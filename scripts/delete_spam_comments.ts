/**
 * Delete Spam Comments Script
 * Uses AWS Bedrock to semantically detect and delete spam comments on GitHub issues.
 */

import { Octokit } from "@octokit/rest";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { retryWithBackoff } from "./retry_utils.js";
import { checkRateLimit, processBatch } from "./rate_limit_utils.js";

const BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0";
const CONFIDENCE_THRESHOLD = 0.85;

export interface SpamCheckResult {
  isSpam: boolean;
  reason: string;
  confidence: number;
}

function createBedrockClient(): BedrockRuntimeClient {
  return new BedrockRuntimeClient({
    region: process.env.AWS_REGION || "us-east-1",
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
    },
  });
}

/**
 * Sanitize comment body to prevent prompt injection.
 * Strips null bytes and limits length; content is passed as a separate user
 * message (never interpolated into the system prompt) so injection is not
 * structurally possible, but we still normalise the input defensively.
 */
function sanitizeCommentBody(body: string): string {
  return body
    .replace(/\0/g, "") // strip null bytes
    .substring(0, 2000)  // hard cap — model doesn't need more
    .trim();
}

/**
 * Use Bedrock to semantically detect spam, including obfuscated/homoglyph content.
 * The system prompt is required via the SPAM_DETECTION_PROMPT env var.
 * The comment body is passed as a separate user message so it can never
 * override or escape the system instructions.
 */
export async function isSpamComment(body: string): Promise<SpamCheckResult> {
  if (!body.trim()) {
    return { isSpam: false, reason: "Empty comment", confidence: 0 };
  }

  const systemPrompt = process.env.SPAM_DETECTION_PROMPT;
  if (!systemPrompt) {
    throw new Error("Missing required environment variable: SPAM_DETECTION_PROMPT");
  }

  const client = createBedrockClient();

  // Sanitize and isolate the comment — never interpolate into the system prompt.
  const safeBody = sanitizeCommentBody(body);

  try {
    const responseBody = await retryWithBackoff(async () => {
      const command = new InvokeModelCommand({
        modelId: BEDROCK_MODEL_ID,
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 256,
          temperature: 0.1,
          system: systemPrompt,
          // Comment is the sole user message — structurally isolated from instructions.
          messages: [{ role: "user", content: safeBody }],
        }),
      });
      const response = await client.send(command);
      return new TextDecoder().decode(response.body);
    });

    const parsed = JSON.parse(responseBody);
    const text = parsed.content?.find((c: any) => c.type === "text")?.text ?? "";
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error("No JSON in Bedrock response");

    const result = JSON.parse(jsonMatch[0]);
    const confidence = result.confidence ?? 0;
    const isSpam = result.is_spam === true && confidence >= CONFIDENCE_THRESHOLD;

    return { isSpam, reason: result.reason ?? "", confidence };
  } catch (err) {
    console.warn("Bedrock spam check failed, skipping:", err);
    return { isSpam: false, reason: "Bedrock check failed", confidence: 0 };
  }
}

async function deleteComment(
  client: Octokit,
  owner: string,
  repo: string,
  commentId: number
): Promise<void> {
  await retryWithBackoff(async () => {
    await client.issues.deleteComment({ owner, repo, comment_id: commentId });
  });
}

async function processSingleComment(
  client: Octokit,
  owner: string,
  repo: string,
  commentId: number,
  commentBody: string,
  commentAuthor: string
): Promise<boolean> {
  console.log(`Checking comment #${commentId} by @${commentAuthor}...`);
  const result = await isSpamComment(commentBody);

  if (!result.isSpam) {
    console.log(`Clean (confidence: ${result.confidence.toFixed(2)}). Reason: ${result.reason}`);
    return false;
  }

  console.log(`Spam detected (confidence: ${result.confidence.toFixed(2)}). Reason: ${result.reason}`);
  await deleteComment(client, owner, repo, commentId);
  console.log(`Deleted spam comment #${commentId}`);
  return true;
}

async function bulkScanAndDelete(
  client: Octokit,
  owner: string,
  repo: string
): Promise<{ scanned: number; deleted: number }> {
  console.log(`Starting bulk spam scan for ${owner}/${repo}...`);

  let scanned = 0;
  let deleted = 0;
  let page = 1;

  while (true) {
    await checkRateLimit(client);

    const { data: comments } = await retryWithBackoff(() =>
      client.issues.listCommentsForRepo({
        owner,
        repo,
        per_page: 100,
        page,
        sort: "created",
        direction: "desc",
      })
    );

    if (comments.length === 0) break;

    console.log(`Processing page ${page} (${comments.length} comments)...`);

    const results = await processBatch(
      comments,
      5, // smaller batch size — each item makes a Bedrock API call
      async (comment) => {
        scanned++;
        const body = comment.body ?? "";
        const author = comment.user?.login ?? "unknown";
        const result = await isSpamComment(body);

        if (result.isSpam) {
          console.log(`Spam in comment #${comment.id} by @${author}: ${result.reason}`);
          try {
            await deleteComment(client, owner, repo, comment.id);
            console.log(`Deleted comment #${comment.id}`);
            return true;
          } catch (err) {
            console.error(`Failed to delete comment #${comment.id}:`, err);
            return false;
          }
        }
        return false;
      },
      1000 // 1s delay between batches to respect Bedrock rate limits
    );

    deleted += results.filter(Boolean).length;
    page++;
  }

  return { scanned, deleted };
}

async function main() {
  const owner = process.env.REPOSITORY_OWNER || "";
  const repo = process.env.REPOSITORY_NAME || "";
  const githubToken = process.env.GITHUB_TOKEN || "";
  const commentId = process.env.COMMENT_ID ? parseInt(process.env.COMMENT_ID) : null;
  const commentBody = process.env.COMMENT_BODY ?? "";
  const commentAuthor = process.env.COMMENT_AUTHOR ?? "unknown";
  const mode = process.env.SCAN_MODE || (commentId ? "single" : "bulk");

  if (!owner || !repo || !githubToken) {
    console.error("Missing required environment variables: REPOSITORY_OWNER, REPOSITORY_NAME, GITHUB_TOKEN");
    process.exit(1);
  }

  if (!process.env.SPAM_DETECTION_PROMPT) {
    console.error("Missing required environment variable: SPAM_DETECTION_PROMPT");
    process.exit(1);
  }

  const client = new Octokit({ auth: githubToken });

  if (mode === "single" && commentId) {
    console.log(`=== Single Comment Spam Check (comment #${commentId}) ===`);
    const deleted = await processSingleComment(client, owner, repo, commentId, commentBody, commentAuthor);
    console.log(`\nSummary: ${deleted ? `Spam comment #${commentId} by @${commentAuthor} was deleted.` : `Comment #${commentId} is clean — no action taken.`}`);
  } else {
    console.log(`=== Bulk Spam Scan for ${owner}/${repo} ===`);
    const { scanned, deleted } = await bulkScanAndDelete(client, owner, repo);
    console.log(`\nSummary: Scanned ${scanned} comments, deleted ${deleted} spam comments.`);
  }
}

// Only run when executed directly (not when imported by tests)
if (process.env.JEST_WORKER_ID === undefined) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}
