import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import { GlmParser } from "../src/glm-parser";
import { resolveModelPath } from "../src/model_path";
import { DefaultChatModelParser, type OutputParserEvent } from "../src/chat-model-parser";
import { glmParserCorpus } from "./glm_parser_corpus";

type LoadedTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

describe("GlmParser", () => {
  let tokenizer: LoadedTokenizer;

  before(async () => {
    const tokenizerDir = resolveModelPath("lukealonso/GLM-5.2-NVFP4");
    tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });
  });

  for (const sample of glmParserCorpus) {
    it(`parses model output token by token: ${sample.name}`, () => {
      const parser = new GlmParser(tokenizer);
      const tokenIds = tokenizer.encode(sample.output, { add_special_tokens: false });
      const events: OutputParserEvent[] = [];

      for (const tokenId of tokenIds) events.push(...parser.onToken(tokenId));
      events.push(...parser.finish());

      const errors = events.filter(event => event.type === "parse_error");
      assert.deepEqual(errors, []);

      const reasoning = events
        .filter(event => event.type === "reasoning_delta")
        .map(event => event.text)
        .join("");
      const content = events
        .filter(event => event.type === "content_delta")
        .map(event => event.text)
        .join("");
      const toolCalls = events
        .filter(event => event.type === "tool_call")
        .map(event => ({
          name: event.name,
          arguments: JSON.parse(event.arguments) as Record<string, unknown>,
        }));

      assert.equal(reasoning, sample.reasoning);
      assert.equal(content, sample.content);
      assert.deepEqual(toolCalls, sample.toolCalls);
      assert.equal(parser.producedToolCalls, sample.toolCalls.length > 0);
      assert.equal(parser.state, "content");
    });
  }

  it("starts in content mode when thinking is disabled", () => {
    const parser = new GlmParser(tokenizer, { enable_thinking: false });
    const events: OutputParserEvent[] = [];
    const tokenIds = tokenizer.encode("READY", { add_special_tokens: false });

    for (const tokenId of tokenIds) events.push(...parser.onToken(tokenId));
    events.push(...parser.finish());

    assert.deepEqual(events, [{ type: "content_delta", text: "READY" }]);
  });

  it("reports a tool call truncated by the end of generation", () => {
    const parser = new GlmParser(tokenizer);
    const tokenIds = tokenizer.encode(
      "reasoning</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris",
      { add_special_tokens: false },
    );
    const events: OutputParserEvent[] = [];

    for (const tokenId of tokenIds) events.push(...parser.onToken(tokenId));
    events.push(...parser.finish());

    assert.equal(events.at(-1)?.type, "parse_error");
    assert.equal(parser.producedToolCalls, false);
  });

  it("the default parser treats protocol markers as content", () => {
    const parser = new DefaultChatModelParser(tokenizer);
    const output = "reasoning</think><tool_call>get_weather</tool_call>";
    const tokenIds = tokenizer.encode(output, { add_special_tokens: false });
    const events: OutputParserEvent[] = [];

    for (const tokenId of tokenIds) events.push(...parser.onToken(tokenId));
    events.push(...parser.finish());

    assert.equal(
      events
        .filter(event => event.type === "content_delta")
        .map(event => event.text)
        .join(""),
      output,
    );
    assert.equal(events.some(event => event.type !== "content_delta"), false);
  });
});
