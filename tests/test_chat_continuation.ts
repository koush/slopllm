import assert from "node:assert/strict";
import { it } from "node:test";
import { tokenizeContinuation } from "../src/chat-continuation";
import type { Tokenizer } from "../src/chat_model";

it("removes the assistant terminator without adding a generation prompt", () => {
  const tokenizer = {
    apply_chat_template(messages: any[], options: any) {
      assert.equal(options.add_generation_prompt, false);
      assert.equal(options.tokenize, false);
      assert.deepEqual(options.tools, []);
      return `<user>hello</end><assistant>${messages[1].content.trim()}</end>`;
    },
    encode(text: string, options: any) {
      assert.equal(options.add_special_tokens, false);
      assert.equal(text, "<user>hello</end><assistant>The answer is ");
      return [1, 2];
    },
  } as unknown as Tokenizer;
  assert.deepEqual(tokenizeContinuation(tokenizer, [
    { role: "user", content: "hello" }, { role: "assistant", content: "The answer is " },
  ], [], { continue_final_message: true }), [1, 2]);
});

it("rejects unsupported prefixes and conflicting generation prompts", () => {
  const tokenizer = {} as Tokenizer;
  for (const message of [{ role: "user", content: "hello" }, { role: "assistant", content: "" }]) {
    assert.throws(() => tokenizeContinuation(tokenizer, [message], undefined, {}), /final assistant message/);
  }
  assert.throws(() => tokenizeContinuation(tokenizer, [{ role: "assistant", content: "hello" }], undefined,
    { add_generation_prompt: true }), /cannot be combined/);
});
