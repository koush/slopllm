import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import { Template } from "@huggingface/jinja";
import { DEEPSEEK_V41_CHAT_TEMPLATE, DeepseekV41ChatParser } from "../src/deepseek_v41_chat";
import { resolveModelPath } from "../src/model_path";
import { resolveCheckpointConfigDir } from "../src/checkpoint_source";
import type { OutputParserEvent } from "../src/chat-model-parser";

const bos = "<｜begin▁of▁sentence｜>", eos = "<｜end▁of▁sentence｜>";
const template = new Template(DEEPSEEK_V41_CHAT_TEMPLATE);
const render = (messages: Record<string, unknown>[], options: Record<string, unknown> = {}) => template.render({
  messages, bos_token: bos, eos_token: eos, add_generation_prompt: true, ...options,
});

it("V4.1 text prompts match the reference role transitions and thinking prefixes", () => {
  assert.equal(render([{ role: "user", content: "Hello" }], { enable_thinking: false }),
    bos + "<｜User｜>Hello<｜Assistant｜></think>");
  assert.equal(render([{ role: "system", content: "Be brief." }, { role: "user", content: "Hello" }], { reasoning_effort: "none" }),
    bos + "<｜System｜>Be brief.<｜User｜>Hello<｜Assistant｜></think>");
  assert.equal(render([{ role: "user", content: "Hello" }]),
    bos + "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hello<｜Assistant｜><think>");
  const history = [{ role: "user", content: "Hi" }, { role: "assistant", content: "Hello", reasoning_content: "discard me" },
    { role: "user", content: "Part one" }, { role: "user", content: [{ type: "text", text: "Part two" }, { type: "text", text: "Part three" }] }];
  assert.equal(render(history, { reasoning_effort: 1 }), bos + "<｜System｜>Reasoning Effort: 1 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    + "<｜User｜>Hi<｜Assistant｜><think>discard me</think>Hello" + eos + "<｜User｜>Part one\n\nPart two\n\nPart three<｜Assistant｜><think>");
  assert.equal(render(history, { reasoning_effort: 1, drop_thinking: true }), bos + "<｜System｜>Reasoning Effort: 1 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    + "<｜User｜>Hi<｜Assistant｜></think>Hello" + eos + "<｜User｜>Part one\n\nPart two\n\nPart three<｜Assistant｜><think>");
});

it("V4.1 text prompts reject unsupported content and invalid reasoning effort", () => {
  const messages = [{ role: "user", content: "Hello" }];
  for (const effort of [0, 101, 1.5, "medium"]) assert.throws(() => render(messages, { reasoning_effort: effort }), /Invalid V4.1 reasoning_effort/);
  assert.throws(() => render(messages, { tools: [{ type: "function" }] }), /requires function tools/);
  assert.throws(() => render([{ role: "user", content: [{ type: "image_url", image_url: "x" }] }]), /text content only/);
});

const tool = { type: "function", function: { name: "weather", description: "Get weather", parameters: {
  type: "object", properties: { city: { type: "string" } }, required: ["city"],
} } };
const invoke = (name: string, parameters: string) => `<｜DSML｜ invoke name="${name}">\n${parameters}\n</｜DSML｜ invoke>`;
const parameter = (name: string, value: string, string = true) => `<｜DSML｜ parameter name="${name}" string="${string}">${value}</｜DSML｜ parameter>`;
const calls = (...invocations: string[]) => `<｜DSML｜ calls>\n${invocations.join("\n")}\n</｜DSML｜ calls>`;

it("V4.1 renders tool schemas, typed call history, and results in call order", () => {
  for (const thinking of [true, false]) {
    const prompt = render([{ role: "system", content: "Use tools." }, { role: "user", content: "Weather?" }],
      { tools: [tool], enable_thinking: thinking });
    assert.ok(prompt.includes("Use tools.\n\n## Tools"));
    assert.deepEqual(JSON.parse(prompt.split("### Available Tool Schemas\n\n")[1].split("\n")[0]), tool.function);
    assert.ok(prompt.endsWith(thinking ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>"));
  }
  const args = { city: "東京", count: 2, enabled: false, options: { units: "C" }, missing: null, days: [1, 2] };
  const history = [{ role: "user", content: "Weather?" }, { role: "assistant", content: null, tool_calls: [
    { id: "a", type: "function", function: { name: "weather", arguments: args } },
    { id: "b", type: "function", function: { name: "clock", arguments: {} } },
  ] }, { role: "tool", tool_call_id: "b", content: "12:00" },
  { role: "tool", tool_call_id: "a", content: [{ type: "text", text: "Sunny" }] }];
  const prompt = render(history, { tools: [tool], enable_thinking: false });
  assert.ok(prompt.includes("<｜Assistant｜></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"weather\">"));
  assert.ok(prompt.includes(parameter("city", "東京")));
  assert.ok(prompt.includes(parameter("count", "2", false)));
  assert.ok(prompt.includes(invoke("clock", "") + "\n</｜DSML｜ calls>" + eos));
  assert.ok(prompt.endsWith("<｜User｜><tool_result>Sunny</tool_result>\n\n<tool_result>12:00</tool_result><｜Assistant｜></think>"));
});

it("V4.1 tool results preserve current-turn reasoning when dropping earlier thinking", () => {
  const history = [
    { role: "user", content: "Previous question" },
    { role: "assistant", content: "Previous answer", reasoning_content: "Old reasoning" },
    { role: "user", content: "Find the bug" },
    { role: "assistant", content: null, reasoning_content: "Read main first", tool_calls: [
      { id: "a", type: "function", function: { name: "read_file", arguments: { path: "main.ts" } } },
    ] },
    { role: "tool", tool_call_id: "a", content: "Main file contents" },
    { role: "assistant", content: null, reasoning_content: "Check the helper next", tool_calls: [
      { id: "b", type: "function", function: { name: "read_file", arguments: { path: "helper.ts" } } },
    ] },
    { role: "tool", tool_call_id: "b", content: "Helper file contents" },
  ];
  const options = { drop_thinking: true };
  const prompt = render(history, options);
  assert.ok(!prompt.includes("Old reasoning"));
  assert.ok(prompt.includes("<｜User｜>Find the bug<｜Assistant｜><think>Read main first</think>"));
  assert.ok(prompt.includes("<tool_result>Main file contents</tool_result><｜Assistant｜><think>Check the helper next</think>"));
  assert.ok(prompt.endsWith("<tool_result>Helper file contents</tool_result><｜Assistant｜><think>"));
  for (const role of ["user", "system"]) {
    const nextTurn = render([...history, { role, content: "Now fix it" }], options);
    assert.ok(!nextTurn.includes("Read main first"));
    assert.ok(!nextTurn.includes("Check the helper next"));
    assert.ok(nextTurn.endsWith("Now fix it<｜Assistant｜><think>"));
  }
});

describe("V4.1 tool parser", () => {
  let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  before(async () => {
    const dir = resolveCheckpointConfigDir(resolveModelPath("local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF"));
    tokenizer = await AutoTokenizer.from_pretrained(dir, { local_files_only: true });
  });
  const feed = (parser: DeepseekV41ChatParser, text: string) => tokenizer.encode(text, { add_special_tokens: false })
    .flatMap(id => parser.onToken(id));

  it("round-trips typed assistant tool history through the prompt and parser", () => {
    const args = { city: "東京", count: 2, enabled: false, options: { units: "C" }, missing: null, days: [1, 2] };
    const prompt = render([{ role: "user", content: "Weather?" }, { role: "assistant", content: "", tool_calls: [
      { id: "a", type: "function", function: { name: "weather", arguments: args } },
    ] }], { enable_thinking: false });
    const output = prompt.split("<｜Assistant｜></think>")[1].slice(0, -eos.length);
    const parser = new DeepseekV41ChatParser(tokenizer, { enable_thinking: false });
    const events = [...feed(parser, output), ...parser.finish()];
    assert.deepEqual(events.filter(e => e.type === "parse_error"), []);
    assert.deepEqual(events.filter(e => e.type === "tool_call").map(e => JSON.parse(e.arguments)), [args]);
  });

  it("streams multiple DSML calls with exact string and JSON argument types", () => {
    const parser = new DeepseekV41ChatParser(tokenizer, {});
    const events = feed(parser, "Need weather.</think>Checking.\n" + calls(
      invoke("weather", [parameter("city", " 東京 & <tag> \n<think>literal</think>"), parameter("count", "2", false),
        parameter("options", '{"x":[true,null]}', false), parameter("literal", "false")].join("\n")), invoke("clock", "")) + "Done.");
    events.push(...parser.finish());
    assert.deepEqual(events.filter(e => e.type === "parse_error"), []);
    assert.equal(events.filter(e => e.type === "reasoning_delta").map(e => e.text).join(""), "Need weather.");
    assert.equal(events.filter(e => e.type === "content_delta").map(e => e.text).join(""), "Checking.\nDone.");
    assert.deepEqual(events.filter(e => e.type === "tool_call").map(e => ({ index: e.index, name: e.name, arguments: JSON.parse(e.arguments) })), [
      { index: 0, name: "weather", arguments: { city: " 東京 & <tag> \n<think>literal</think>", count: 2, options: { x: [true, null] }, literal: "false" } },
      { index: 1, name: "clock", arguments: {} },
    ]);
    assert.equal(parser.producedToolCalls, true);
    assert.equal(parser.state, "content");
  });

  it("reports malformed and truncated tool calls without emitting them as content", () => {
    for (const text of [calls(invoke("weather", parameter("count", "oops", false))),
      calls(invoke("weather", parameter("x", "a") + parameter("x", "b"))),
      '<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">',
      '<｜DSML｜ calls>bad</｜DSML｜ calls>']) {
      const parser = new DeepseekV41ChatParser(tokenizer, { enable_thinking: false });
      const events = [...feed(parser, text), ...parser.finish()];
      assert.ok(events.some(e => e.type === "parse_error"), text);
      assert.equal(parser.producedToolCalls, false);
      assert.equal(events.filter(e => e.type === "content_delta").map(e => e.text).join(""), "");
    }
  });

  it("resumes a partial invocation and emits each call before the block closes", () => {
    const parser = new DeepseekV41ChatParser(tokenizer, { continue_final_message: true });
    parser.continueFrom(tokenizer.encode('<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">', { add_special_tokens: false }));
    const events = feed(parser, parameter("city", "Paris") + "\n</｜DSML｜ invoke>");
    assert.equal(events.filter(e => e.type === "tool_call").length, 1);
    assert.equal(parser.state, "tool");
    assert.deepEqual(feed(parser, "\n</｜DSML｜ calls>"), []);
    assert.deepEqual(parser.finish(), []);
  });

  it("preserves ordinary text containing partial marker prefixes", () => {
    const parser = new DeepseekV41ChatParser(tokenizer, { enable_thinking: false });
    const text = "東京 <tag> <｜DSML｜ not-a-call> trailing <｜DSML｜ ca";
    const events: OutputParserEvent[] = [...feed(parser, text), ...parser.finish()];
    assert.equal(events.filter(e => e.type === "content_delta").map(e => e.text).join(""), text);
  });
});
