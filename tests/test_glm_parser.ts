import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { before, describe, it } from "node:test";
import { GlmParser } from "../src/glm-parser";
import { Glm51Model } from "../src/glm51_model";
import { resolveModelPath } from "../src/model_path";
import { DefaultChatModelParser, type OutputParserEvent } from "../src/chat-model-parser";
import { glmParserCorpus } from "./glm_parser_corpus";
import { tokenizeContinuation } from "../src/chat-continuation";

type LoadedTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

describe("GlmParser", () => {
  let tokenizer: LoadedTokenizer;

  before(async () => {
    const tokenizerDir = resolveModelPath(process.env.GLM_TOKENIZER_REPO ?? "lukealonso/GLM-5.2-NVFP4");
    tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });
    const templatePath = path.join(tokenizerDir, "chat_template.jinja");
    if (fs.existsSync(templatePath)) {
      tokenizer.chat_template = fs.readFileSync(templatePath, "utf-8").replace(/\.(\d+)\b/g, "[$1]");
    }
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

  it("continues an assistant prefix without returning the prefix as new output", () => {
    const parser = new GlmParser(tokenizer, { continue_final_message: true });
    parser.continueFrom(tokenizer.encode("The answer is", { add_special_tokens: false }));
    const events = tokenizer.encode(" Paris.", { add_special_tokens: false }).flatMap(id => parser.onToken(id));
    events.push(...parser.finish());
    assert.equal(events.map(event => event.type === "content_delta" ? event.text : "").join(""), " Paris.");
    assert.equal(parser.state, "content");
  });

  it("forwards continuation mode through the model parser factory", () => {
    const parser = Glm51Model.prototype.createParser.call({ tokenizer } as Glm51Model, { continue_final_message: true });
    parser.continueFrom(tokenizer.encode("2", { add_special_tokens: false }));
    const events = tokenizer.encode(" + 2 = 4", { add_special_tokens: false }).flatMap(id => parser.onToken(id));
    events.push(...parser.finish());
    assert.ok(events.every(event => event.type === "content_delta"));
    assert.equal(events.map(event => event.type === "content_delta" ? event.text : "").join(""), " + 2 = 4");
  });

  it("renders a continuation with the real GLM chat template", () => {
    const ids = tokenizeContinuation(tokenizer, [
      { role: "user", content: "Name the capital of France." },
      { role: "assistant", content: "The capital is " },
    ], undefined, { continue_final_message: true });
    const text = tokenizer.decode(ids, { skip_special_tokens: false });
    assert.ok(text.endsWith("The capital is "), text);
  });

  it("preserves an open reasoning block in an assistant continuation", () => {
    const parser = new GlmParser(tokenizer, { continue_final_message: true });
    parser.continueFrom(tokenizer.encode("<think>Let me consider", { add_special_tokens: false }));
    assert.equal(parser.state, "reasoning");
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
