import assert from "node:assert/strict";
import { describe, it, before } from "node:test";

const BASE_URL = "http://localhost:8010";
const SERVER_READY_TIMEOUT_MS = 120_000;
const SERVER_READY_POLL_MS = 2_000;
const DEFAULT_MAX_TOKENS = 64;
const FACTUAL_MAX_TOKENS = 512;

const FACTUAL_QUESTIONS: Array<{ question: string; expected: string }> = [
  { question: "What is the capital of France?", expected: "Paris" },
  { question: "What is the capital of Japan?", expected: "Tokyo" },
  { question: "What is the capital of Germany?", expected: "Berlin" },
  { question: "What is the capital of Italy?", expected: "Rome" },
  { question: "What is the capital of Spain?", expected: "Madrid" },
  { question: "What is the largest planet in the solar system?", expected: "Jupiter" },
  { question: "What color is the sky on a clear day?", expected: "blue" },
  { question: "How many days are in a week?", expected: "7" },
  { question: "What is the chemical symbol for water?", expected: "H2O" },
  { question: "What is 1 plus 1?", expected: "2" },
  { question: "What is 2 plus 3?", expected: "5" },
  { question: "What is the capital of the United Kingdom?", expected: "London" },
  { question: "What is the capital of China?", expected: "Beijing" },
  { question: "What gas do plants absorb from the atmosphere?", expected: "arbon dioxide" },
  { question: "What is the boiling point of water in Celsius?", expected: "100" },
  { question: "What is the capital of Australia?", expected: "Canberra" },
];

function stripThinking(text: string): string {
  const endTag = "</think>";
  const idx = text.indexOf(endTag);
  if (idx === -1) return text;
  return text.slice(idx + endTag.length).trim();
}

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

async function chatCompletion(body: Record<string, unknown>): Promise<Response> {
  return fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function chatCompletionText(body: Record<string, unknown>): Promise<{ content: string; answer: string; finishReason: string; usage: Record<string, number> }> {
  const res = await chatCompletion(body);
  assert.equal(res.status, 200);
  const data = await res.json() as Record<string, unknown>;
  const choice = (data.choices as Array<Record<string, unknown>>)[0];
  const message = choice.message as Record<string, unknown>;
  const content = message.content as string;
  return {
    content,
    answer: stripThinking(content),
    finishReason: choice.finish_reason as string,
    usage: data.usage as Record<string, number>,
  };
}

async function collectSSE(response: Response): Promise<Array<Record<string, unknown>>> {
  const events: Array<Record<string, unknown>> = [];
  const text = await response.text();
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data: ")) continue;
    const data = trimmed.slice(6);
    if (data === "[DONE]") break;
    try {
      events.push(JSON.parse(data));
    } catch {
      // skip malformed
    }
  }
  return events;
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

function greedyOverrides(): Record<string, unknown> {
  return { temperature: 0, top_k: 1 };
}

describe("OpenAI server", () => {
  before(async () => {
    await waitForServer();
  });

  // --- Endpoint tests ---

  it("health endpoint returns ok", async () => {
    const res = await fetch(`${BASE_URL}/health`);
    assert.equal(res.status, 200);
    const data = await res.json() as Record<string, unknown>;
    assert.equal(data.status, "ok");
    assert.equal(data.model, "qwen3-0.6b");
  });

  it("models endpoint returns model list", async () => {
    const res = await fetch(`${BASE_URL}/v1/models`);
    assert.equal(res.status, 200);
    const data = await res.json() as Record<string, unknown>;
    assert.equal(data.object, "list");
    const models = data.data as Array<Record<string, unknown>>;
    assert.ok(models.length >= 1);
    assert.equal(models[0].id, "qwen3-0.6b");
    assert.equal(models[0].object, "model");
  });

  it("unknown route returns 404", async () => {
    const res = await fetch(`${BASE_URL}/unknown`);
    assert.equal(res.status, 404);
  });

  // --- Non-streaming completion ---

  it("non-streaming completion returns valid response format", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("What is 2+2?"),
      max_tokens: DEFAULT_MAX_TOKENS,
      stream: false,
    });
    assert.equal(res.status, 200);
    const data = await res.json() as Record<string, unknown>;
    assert.equal(data.object, "chat.completion");
    assert.ok(data.id && typeof data.id === "string" && data.id.startsWith("chatcmpl-"));
    assert.ok(data.created && typeof data.created === "number");
    assert.equal(data.model, "qwen3-0.6b");
    const choices = data.choices as Array<Record<string, unknown>>;
    assert.equal(choices.length, 1);
    assert.equal(choices[0].index, 0);
    const message = choices[0].message as Record<string, unknown>;
    assert.equal(message.role, "assistant");
    assert.ok(typeof message.content === "string" && message.content.length > 0);
    assert.ok(typeof choices[0].finish_reason === "string");
    const usage = data.usage as Record<string, number>;
    assert.ok(usage.prompt_tokens > 0);
    assert.ok(usage.completion_tokens > 0);
    assert.ok(usage.total_tokens === usage.prompt_tokens + usage.completion_tokens);
  });

  // --- Content validation (with thinking stripped) ---

  it("factual question: capital of France", async () => {
    const { answer } = await chatCompletionText({
      messages: simpleMessages("What is the capital of France?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
    });
    assert.ok(containsIgnoreCase(answer, "Paris"), `Expected "Paris" in answer, got: "${answer.slice(0, 200)}"`);
  });

  it("factual question: capital of Japan", async () => {
    const { answer } = await chatCompletionText({
      messages: simpleMessages("What is the capital of Japan?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
    });
    assert.ok(containsIgnoreCase(answer, "Tokyo"), `Expected "Tokyo" in answer, got: "${answer.slice(0, 200)}"`);
  });

  it("factual question: largest planet", async () => {
    const { answer } = await chatCompletionText({
      messages: simpleMessages("What is the largest planet in the solar system?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
    });
    assert.ok(containsIgnoreCase(answer, "Jupiter"), `Expected "Jupiter" in answer, got: "${answer.slice(0, 200)}"`);
  });

  it("factual question: simple arithmetic", async () => {
    const { answer } = await chatCompletionText({
      messages: simpleMessages("What is 2 plus 3?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
    });
    assert.ok(containsIgnoreCase(answer, "5"), `Expected "5" in answer, got: "${answer.slice(0, 200)}"`);
  });

  it("factual question: days in a week", async () => {
    const { answer } = await chatCompletionText({
      messages: simpleMessages("How many days are in a week?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
    });
    assert.ok(answer.includes("7"), `Expected "7" in answer, got: "${answer.slice(0, 200)}"`);
  });

  // --- Streaming ---

  it("streaming completion returns SSE chunks", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("Say hello"),
      max_tokens: DEFAULT_MAX_TOKENS,
      stream: true,
    });
    assert.equal(res.status, 200);
    const contentType = res.headers.get("content-type") ?? "";
    assert.ok(contentType.includes("text/event-stream"), `Expected text/event-stream, got ${contentType}`);

    const events = await collectSSE(res);
    assert.ok(events.length >= 2, `Expected at least 2 events, got ${events.length}`);

    const first = events[0];
    assert.equal(first.object, "chat.completion.chunk");
    const firstChoices = first.choices as Array<Record<string, unknown>> | undefined;
    const firstDelta = firstChoices?.[0];
    assert.ok(firstDelta, "First chunk missing choices");
    const firstContent = (firstDelta.delta as Record<string, unknown>)?.content;
    assert.equal(firstContent, "", "First delta should have empty content (role only)");

    let hasContent = false;
    let finishReason: string | null = null;
    for (const event of events) {
      const choice = (event.choices as Array<Record<string, unknown>> | undefined)?.[0];
      if (!choice) continue;
      const delta = choice.delta as Record<string, unknown> | undefined;
      if (delta?.content && (delta.content as string).length > 0) hasContent = true;
      if (choice.finish_reason) finishReason = choice.finish_reason as string;
    }
    assert.ok(hasContent, "Expected at least one chunk with content");
    assert.ok(finishReason, "Expected finish_reason in final chunk");
  });

  it("streaming completion ends with data: [DONE]", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("Hi"),
      max_tokens: 16,
      stream: true,
    });
    const text = await res.text();
    assert.ok(text.includes("data: [DONE]"), "Expected data: [DONE] terminator");
  });

  it("streaming with include_usage returns usage data", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("Hi"),
      max_tokens: 16,
      stream: true,
      stream_options: { include_usage: true },
    });
    const events = await collectSSE(res);
    const usageEvent = events.find(e => {
      const choices = e.choices as Array<unknown>;
      return Array.isArray(choices) && choices.length === 0 && e.usage;
    });
    assert.ok(usageEvent, "Expected a chunk with choices:[] and usage");
    const usage = usageEvent!.usage as Record<string, number>;
    assert.ok(usage.prompt_tokens > 0);
    assert.ok(usage.completion_tokens > 0);
  });

  it("streaming factual content validation", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("What is the capital of France?"),
      max_tokens: FACTUAL_MAX_TOKENS,
      ...greedyOverrides(),
      stream: true,
    });
    assert.equal(res.status, 200);
    const events = await collectSSE(res);
    let fullContent = "";
    for (const event of events) {
      const choice = (event.choices as Array<Record<string, unknown>> | undefined)?.[0];
      if (!choice) continue;
      const delta = choice.delta as Record<string, unknown> | undefined;
      if (delta?.content) fullContent += delta.content as string;
    }
    const answer = stripThinking(fullContent);
    assert.ok(containsIgnoreCase(answer, "Paris"), `Expected "Paris" in streamed answer, got: "${answer.slice(0, 200)}"`);
  });

  // --- Streaming vs non-streaming parity ---

  it("streaming and non-streaming produce same content with greedy decoding", async () => {
    const messages = simpleMessages("What is 1+1?");

    const nonStreaming = await chatCompletionText({ messages, max_tokens: FACTUAL_MAX_TOKENS, ...greedyOverrides() });

    const res = await chatCompletion({ messages, max_tokens: FACTUAL_MAX_TOKENS, ...greedyOverrides(), stream: true });
    const events = await collectSSE(res);
    let streamingContent = "";
    for (const event of events) {
      const choice = (event.choices as Array<Record<string, unknown>> | undefined)?.[0];
      if (!choice) continue;
      const delta = choice.delta as Record<string, unknown> | undefined;
      if (delta?.content) streamingContent += delta.content as string;
    }

    assert.equal(stripThinking(streamingContent).trim(), stripThinking(nonStreaming.content).trim(),
      `Streaming answer "${stripThinking(streamingContent).trim()}" !== non-streaming "${stripThinking(nonStreaming.content).trim()}"`);
  });

  // --- Greedy reproducibility ---

  it("greedy decoding is reproducible", async () => {
    const messages = simpleMessages("What is the capital of Germany?");
    const first = await chatCompletionText({ messages, max_tokens: FACTUAL_MAX_TOKENS, ...greedyOverrides() });
    const second = await chatCompletionText({ messages, max_tokens: FACTUAL_MAX_TOKENS, ...greedyOverrides() });
    assert.equal(first.content.trim(), second.content.trim(),
      `Greedy responses differ:\n  first:  "${first.content.trim().slice(0, 200)}"\n  second: "${second.content.trim().slice(0, 200)}"`);
  });

  // --- Parameter tests ---

  it("max_tokens is respected", async () => {
    const maxTokens = 3;
    const res = await chatCompletion({
      messages: simpleMessages("Write a very long essay about the history of computing"),
      max_tokens: maxTokens,
      stream: false,
    });
    assert.equal(res.status, 200);
    const data = await res.json() as Record<string, unknown>;
    const usage = data.usage as Record<string, number>;
    assert.ok(
      usage.completion_tokens <= maxTokens,
      `completion_tokens ${usage.completion_tokens} exceeds max_tokens ${maxTokens}`,
    );
  });

  it("stop sequences work", async () => {
    const res = await chatCompletion({
      messages: simpleMessages("Count: 1, 2, 3, 4, 5"),
      max_tokens: 64,
      stream: false,
      stop: [","],
    });
    assert.equal(res.status, 200);
    const data = await res.json() as Record<string, unknown>;
    const content = (data.choices as Array<Record<string, unknown>>)[0].message as Record<string, unknown>;
    const text = content.content as string;
    assert.ok(!text.includes(","), `Response should not contain stop sequence ',', got: "${text.slice(0, 100)}"`);
  });

  it("invalid JSON body returns 400", async () => {
    const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json",
    });
    assert.equal(res.status, 400);
  });

  it("missing messages returns 400", async () => {
    const res = await chatCompletion({ max_tokens: 10 });
    assert.equal(res.status, 400);
  });

  // --- Concurrency tests with content validation ---

  it("batch concurrency: 4 simultaneous requests with different factual questions", async () => {
    const questions = FACTUAL_QUESTIONS.slice(0, 4);
    const requests = questions.map(({ question, expected }, i) =>
      chatCompletion({
        messages: simpleMessages(question),
        max_tokens: FACTUAL_MAX_TOKENS,
        ...greedyOverrides(),
        stream: false,
      }).then(async res => {
        assert.equal(res.status, 200, `Request ${i} (${question}) failed`);
        const data = await res.json() as Record<string, unknown>;
        const content = ((data.choices as Array<Record<string, unknown>>)[0].message as Record<string, unknown>).content as string;
        const answer = stripThinking(content);
        assert.ok(answer.length > 0, `Request ${i} (${question}) returned empty answer`);
        assert.ok(containsIgnoreCase(answer, expected),
          `Request ${i} ("${question}"): expected "${expected}" in "${answer.slice(0, 200)}"`);
      }),
    );
    await Promise.all(requests);
  });

  it("queue beyond batch-size: 6 simultaneous requests with different questions", async () => {
    const questions = FACTUAL_QUESTIONS.slice(4, 10);
    const requests = questions.map(({ question, expected }, i) =>
      chatCompletion({
        messages: simpleMessages(question),
        max_tokens: FACTUAL_MAX_TOKENS,
        ...greedyOverrides(),
        stream: false,
      }).then(async res => {
        assert.equal(res.status, 200, `Queued request ${i} (${question}) failed`);
        const data = await res.json() as Record<string, unknown>;
        const content = ((data.choices as Array<Record<string, unknown>>)[0].message as Record<string, unknown>).content as string;
        const answer = stripThinking(content);
        assert.ok(answer.length > 0, `Queued request ${i} (${question}) returned empty answer`);
        assert.ok(containsIgnoreCase(answer, expected),
          `Queued request ${i} ("${question}"): expected "${expected}" in "${answer.slice(0, 200)}"`);
      }),
    );
    await Promise.all(requests);
  });

  it("concurrent streaming: 3 simultaneous streaming requests with different questions", async () => {
    const questions = FACTUAL_QUESTIONS.slice(10, 13);
    const requests = questions.map(({ question, expected }, i) =>
      chatCompletion({
        messages: simpleMessages(question),
        max_tokens: FACTUAL_MAX_TOKENS,
        ...greedyOverrides(),
        stream: true,
      }).then(async res => {
        assert.equal(res.status, 200, `Streaming request ${i} (${question}) failed`);
        const events = await collectSSE(res);
        let fullContent = "";
        let finishReason: string | null = null;
        for (const event of events) {
          const choice = (event.choices as Array<Record<string, unknown>> | undefined)?.[0];
          if (!choice) continue;
          const delta = choice.delta as Record<string, unknown> | undefined;
          if (delta?.content) fullContent += delta.content as string;
          if (choice.finish_reason) finishReason = choice.finish_reason as string;
        }
        const answer = stripThinking(fullContent);
        assert.ok(answer.length > 0, `Streaming request ${i} (${question}) had no answer content`);
        assert.ok(finishReason, `Streaming request ${i} (${question}) had no finish_reason`);
        assert.ok(containsIgnoreCase(answer, expected),
          `Streaming request ${i} ("${question}"): expected "${expected}" in "${answer.slice(0, 200)}"`);
      }),
    );
    await Promise.all(requests);
  });

  // --- Client disconnect ---

  it("client disconnect during streaming doesn't crash server", async () => {
    const controller = new AbortController();
    const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: simpleMessages("Tell me a long story"),
        max_tokens: 256,
        stream: true,
      }),
      signal: controller.signal,
    });

    const reader = res.body?.getReader();
    assert.ok(reader, "Expected readable stream");
    let chunks = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks++;
        if (chunks >= 3) {
          controller.abort();
          break;
        }
      }
    } catch (e: unknown) {
      assert.ok(
        e instanceof DOMException || (e instanceof Error && e.name === "AbortError"),
        `Expected AbortError, got: ${e}`,
      );
    }
    reader.cancel().catch(() => {});

    await new Promise(r => setTimeout(r, 500));

    const healthRes = await fetch(`${BASE_URL}/health`);
    assert.equal(healthRes.status, 200);
  });
});
