import assert from "node:assert/strict";
import { it } from "node:test";
import { Template } from "@huggingface/jinja";
import { DEEPSEEK_V41_CHAT_TEMPLATE } from "../src/deepseek_v41_chat";

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
    + "<｜User｜>Hi<｜Assistant｜></think>Hello" + eos + "<｜User｜>Part one\n\nPart two\n\nPart three<｜Assistant｜><think>");
});

it("V4.1 text prompts reject unsupported content and invalid reasoning effort", () => {
  const messages = [{ role: "user", content: "Hello" }];
  for (const effort of [0, 101, 1.5, "medium"]) assert.throws(() => render(messages, { reasoning_effort: effort }), /Invalid V4.1 reasoning_effort/);
  assert.throws(() => render(messages, { tools: [{ type: "function" }] }), /does not support tools/);
  assert.throws(() => render([{ role: "user", content: [{ type: "image_url", image_url: "x" }] }]), /text content only/);
});
