import assert from "node:assert/strict";
import { describe, it, before, after } from "node:test";

const BASE_URL = "http://localhost:8010";
const SERVER_READY_TIMEOUT_MS = 120_000;
const SERVER_READY_POLL_MS = 2_000;

async function waitForServer(): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < SERVER_READY_TIMEOUT_MS) {
    try {
      const res = await fetch(`${BASE_URL}/health`);
      if (res.ok) return;
    } catch {}
    await new Promise(r => setTimeout(r, SERVER_READY_POLL_MS));
  }
  throw new Error(`Server not ready after ${SERVER_READY_TIMEOUT_MS / 1000}s`);
}

function stripThinking(text: string): string {
  const endTag = "</think>";
  const idx = text.indexOf(endTag);
  if (idx === -1) return text;
  return text.slice(idx + endTag.length).trim();
}

function simpleMessages(content: string): Array<{ role: string; content: string }> {
  return [{ role: "user", content }];
}

const SUBSCRIPT_MAP = new Map<string, string>([
  ["\u2080", "0"], ["\u2081", "1"], ["\u2082", "2"], ["\u2083", "3"], ["\u2084", "4"],
  ["\u2085", "5"], ["\u2086", "6"], ["\u2087", "7"], ["\u2088", "8"], ["\u2089", "9"],
]);
const SUBSCRIPT_RE = /[\u2080-\u2089]/g;

function containsIgnoreCase(haystack: string, needle: string): boolean {
  const normalized = haystack.replace(SUBSCRIPT_RE, ch => SUBSCRIPT_MAP.get(ch) ?? ch);
  return normalized.toLowerCase().includes(needle.toLowerCase());
}

async function chatCompletion(body: Record<string, unknown>): Promise<Response> {
  return fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function chatCompletionTimed(body: Record<string, unknown>): Promise<{ content: string; answer: string; elapsedMs: number }> {
  const start = Date.now();
  const res = await chatCompletion(body);
  assert.equal(res.status, 200);
  const data = await res.json() as Record<string, unknown>;
  const choice = (data.choices as Array<Record<string, unknown>>)[0];
  const message = choice.message as Record<string, unknown>;
  const content = message.content as string;
  const elapsedMs = Date.now() - start;
  return { content, answer: stripThinking(content), elapsedMs };
}

describe("Continuous batching", () => {
  before(async () => {
    await waitForServer();
  });

  it("concurrent requests finish at similar times", async () => {
    const questions = [
      "What is the capital of France?",
      "What is the capital of Japan?",
      "What is the capital of Germany?",
    ];

    // Send all 3 requests with streaming, track first-token and finish times
    const startTimes: number[] = [];
    const endTimes: number[] = [];
    const answers: string[] = [];

    const requests = questions.map(q =>
      chatCompletion({
        messages: simpleMessages(q),
        max_tokens: 64,
        temperature: 0,
        top_k: 1,
        stream: true,
      }).then(async res => {
        let firstTokenTime = 0;
        let fullContent = "";
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          for (const line of text.split("\n")) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;
            try {
              const event = JSON.parse(data) as Record<string, unknown>;
              const choice = (event.choices as Array<Record<string, unknown>> | undefined)?.[0];
              const delta = choice?.delta as Record<string, unknown> | undefined;
              if (delta?.content && !firstTokenTime) {
                firstTokenTime = Date.now();
              }
              if (delta?.content) fullContent += delta.content as string;
            } catch {}
          }
        }
        const endTime = Date.now();
        if (!firstTokenTime) firstTokenTime = endTime;
        startTimes.push(firstTokenTime);
        endTimes.push(endTime);
        answers.push(stripThinking(fullContent));
      })
    );

    await Promise.all(requests);

    for (let i = 0; i < answers.length; i++) {
      assert.ok(answers[i].length > 0, `Request ${i} returned empty answer`);
    }

    const maxStart = Math.max(...startTimes);
    const minEnd = Math.min(...endTimes);

    // If concurrent, all requests overlap: max(first-token time) < min(finish time).
    // If sequential, request 2 starts after request 1 finishes: max(start) >= min(end).
    assert.ok(maxStart < minEnd,
      `Requests not concurrent: max start ${maxStart} >= min end ${minEnd} ` +
      `(starts: ${startTimes.join(", ")}, ends: ${endTimes.join(", ")})`);
  });

  it("late-joining request gets prefilled mid-generation", async () => {
    // Send request A with streaming, wait for first token, then send request B.
    // Request B should start generating while A is still going.
    const startA = Date.now();
    const resA = await chatCompletion({
      messages: simpleMessages("Count from 1 to 10."),
      max_tokens: 80,
      temperature: 0,
      top_k: 1,
      stream: true,
    });

    // Wait for first SSE chunk to prove decode has started
    const readerA = resA.body!.getReader();
    const decoder = new TextDecoder();
    let firstChunkTime = 0;
    while (true) {
      const { done, value } = await readerA.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      if (text.includes("data: ") && !text.includes("[DONE]")) {
        firstChunkTime = Date.now();
        break;
      }
    }

    // Now send request B while A is still streaming
    assert.ok(firstChunkTime > 0, "Should have received first chunk from A");
    const resultB = await chatCompletionTimed({
      messages: simpleMessages("Count from 1 to 5."),
      max_tokens: 60,
      temperature: 0,
      top_k: 1,
    });

    assert.ok(resultB.answer.length > 0, "Request B returned empty answer");
    assert.ok(containsIgnoreCase(resultB.answer, "1"),
      `Expected "1" in response (answer: "${resultB.answer.slice(0, 200)}")`);

    // Drain the rest of A's stream
    while (true) {
      const { done } = await readerA.read();
      if (done) break;
    }

    const elapsedA = Date.now() - startA;
    // A should still be going when B starts, and both should finish reasonably quickly
    // With continuous batching, B gets prefilled during A's decode loop
    assert.ok(elapsedA < 15000, `Request A took too long: ${elapsedA}ms`);
  });
});
