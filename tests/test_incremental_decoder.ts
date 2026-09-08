import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import { DefaultChatModelParser, type OutputParserEvent } from "../src/chat-model-parser";
import type { Tokenizer } from "../src/chat_model";
import { GlmParser } from "../src/glm-parser";
import { resolveModelSelection } from "../src/model_cli";
import { resolveModelPath } from "../src/model_path";
import { glmParserCorpus } from "./glm_parser_corpus";

// Frozen full-prefix algorithm: compare each emission, including flush, not
// just concatenated text (which can hide premature replacement characters).
class FullPrefixDecoder {
  private ids: number[] = [];
  private emitted = "";

  constructor(private readonly tokenizer: Tokenizer) {}

  push(id: number): string {
    this.ids.push(id);
    const text = this.tokenizer.decode(this.ids, { skip_special_tokens: false });
    let end = text.length;
    while (end > 0 && text.charCodeAt(end - 1) === 0xFFFD) end--;
    const safe = text.slice(0, end);
    const delta = this.delta(safe);
    this.emitted = safe;
    return delta;
  }

  flush(): string {
    if (!this.ids.length) return "";
    const delta = this.delta(this.tokenizer.decode(this.ids, { skip_special_tokens: false }));
    this.ids = [];
    this.emitted = "";
    return delta;
  }

  private delta(text: string): string {
    if (text.startsWith(this.emitted)) return text.slice(this.emitted.length);
    let common = 0;
    while (common < this.emitted.length && common < text.length && this.emitted[common] === text[common]) common++;
    return text.slice(common);
  }
}

function events(text: string): OutputParserEvent[] {
  return text ? [{ type: "content_delta", text }] : [];
}

describe("incremental token history compaction (CPU)", () => {
  let tokenizer: Tokenizer;
  let byteIds: number[];
  let addedIds: number[];
  const encode = (text: string) => tokenizer.encode(text, { add_special_tokens: false });

  before(async () => {
    const modelDir = process.env.GLM_TOKENIZER_REPO
      ? resolveModelPath(process.env.GLM_TOKENIZER_REPO)
      : resolveModelSelection({
        useGlm51: true, useQwen35: false, useFp8: false, useNvfp4: true,
        glm51Small: false, modelDir: undefined, gpus: [], arena: 0, cp: false, mtp: false,
      }).modelDir;
    tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });
    const internal = tokenizer as unknown as {
      _tokenizerJSON: { decoder: { type: string }; model: { vocab: Record<string, number> }; added_tokens: { id: number }[] };
      _tokenizer: { clean_up_tokenization_spaces: boolean };
    };
    assert.equal(internal._tokenizerJSON.decoder.type, "ByteLevel");
    assert.equal(internal._tokenizer.clean_up_tokenization_spaces, false);
    // GPT-2's reversible byte alphabet lets us force splits independently of BPE.
    const bytes = Array.from({ length: 256 }, (_, i) => i).filter(i =>
      (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174);
    const chars = [...bytes];
    for (let i = 0, n = 0; i < 256; i++) {
      if (!bytes.includes(i)) {
        bytes.push(i);
        chars.push(256 + n++);
      }
    }
    byteIds = [];
    bytes.forEach((byte, i) => {
      const id = internal._tokenizerJSON.model.vocab[String.fromCharCode(chars[i])];
      assert.equal(typeof id, "number");
      byteIds[byte] = id;
    });
    addedIds = internal._tokenizerJSON.added_tokens.map(token => token.id);
  });

  function check(ids: number[]): void {
    const fast = new DefaultChatModelParser(tokenizer);
    const old = new FullPrefixDecoder(tokenizer);
    ids.forEach((id, i) => assert.deepEqual(fast.onToken(id), events(old.push(id)), `token ${i}: ${id}`));
    assert.deepEqual(fast.finish(), events(old.flush()));
    assert.deepEqual(fast.finish(), events(old.flush()));
    for (const id of encode("after flush")) assert.deepEqual(fast.onToken(id), events(old.push(id)));
    assert.deepEqual(fast.finish(), events(old.flush()));
  }

  it("matches every delta for Unicode, BOMs, replacement literals and malformed bytes", () => {
    for (const text of [
      "ASCII punctuation . isn't cleaned !", "\u4E2D\u6587\u65E5\u672C\u8A9E",
      "\u{1F600}\u{1F469}\u200D\u{1F4BB}e\u0301", "\uFEFFa\uFEFFb\uFEFF",
      "a\uFFFDb\uFFFD\uFFFD", "\uFEFF\uFEFF", "a\u4E2D\u{1F600}\uFEFFz",
    ]) {
      check(encode(text));
      check([...Buffer.from(text)].map(byte => byteIds[byte]));
    }
    for (const bytes of [
      [0xE4], [0xF0, 0x9F], [0x80, 0xBF, 0x61], [0xE4, 0x61, 0xB8, 0xAD],
      [0xED, 0xA0, 0x80], [0xF4, 0x90, 0x80, 0x80], [0xC0, 0xAF],
      [0xEF, 0xBB, 0xBF, 0xEF, 0xBB, 0xBF],
    ]) {
      check(bytes.map(byte => byteIds[byte]));
      check([...encode("anchor"), ...bytes.map(byte => byteIds[byte]), ...encode("end")]);
    }
  });

  it("preserves all added/special tokens as real decoder boundaries", () => {
    assert.ok(addedIds.length);
    for (const id of addedIds) {
      check([...encode("a"), byteIds[0xE4], id, byteIds[0xB8], byteIds[0xAD],
        id, ...[0xEF, 0xBB, 0xBF].map(byte => byteIds[byte]), ...encode("z")]);
    }
  });

  it("retains full history for real ByteLevel BPE end-of-word suffix replacement", () => {
    const TokenizerClass = tokenizer.constructor as new (json: object, config: object) => Tokenizer;
    const withSuffix = new TokenizerClass({
      model: {
        type: "BPE", vocab: { "<": 0, "/": 1, w: 2, ">": 3 },
        merges: [], end_of_word_suffix: "</w>",
      },
      decoder: { type: "ByteLevel" },
      post_processor: null,
      pre_tokenizer: null,
      normalizer: null,
      added_tokens: [],
    }, { clean_up_tokenization_spaces: false });
    const internals = withSuffix as unknown as {
      _tokenizer: { decoder: { end_of_word_suffix: string }; clean_up_tokenization_spaces: boolean };
    };
    assert.equal(internals._tokenizer.decoder.end_of_word_suffix, "</w>");
    assert.equal(internals._tokenizer.clean_up_tokenization_spaces, false);
    const parser = new DefaultChatModelParser(withSuffix);
    const old = new FullPrefixDecoder(withSuffix);
    for (const [id, expected] of ["<", "/", "w", " "].entries()) {
      const delta = old.push(id);
      assert.equal(delta, expected);
      assert.deepEqual(parser.onToken(id), events(delta));
    }
    assert.deepEqual(parser.finish(), events(old.flush()));
  });

  it("preserves long fragment-only output without promising bounded history", () => {
    const fragment = encode("\u{20000}");
    assert.ok(fragment.length > 1);
    assert.ok(fragment.every(id => tokenizer.decode([id], { skip_special_tokens: false }).includes("\uFFFD")));
    const lengths: number[] = [];
    const measured = new Proxy(tokenizer, {
      get(target, key, receiver) {
        if (key === "decode") return (ids: number[], options: Parameters<Tokenizer["decode"]>[1]) => {
          lengths.push(ids.length);
          return target.decode(ids, options);
        };
        return Reflect.get(target, key, receiver);
      },
    });
    const ids = Array.from({ length: 512 }, () => fragment).flat();
    const parser = new DefaultChatModelParser(measured);
    const old = new FullPrefixDecoder(tokenizer);
    let output = "";
    for (const id of ids) {
      const delta = old.push(id);
      assert.deepEqual(parser.onToken(id), events(delta));
      output += delta;
    }
    const last = old.flush();
    assert.deepEqual(parser.finish(), events(last));
    assert.equal(output + last, "\u{20000}".repeat(512));
    assert.equal(Math.max(...lengths), ids.length);
  });

  it("matches full GlmParser events, state, continuation and finish", () => {
    const fallback = new Proxy(tokenizer, {
      get(target, key, receiver) {
        return key === "_tokenizerJSON" ? undefined : Reflect.get(target, key, receiver);
      },
    });
    const samples = [
      ...glmParserCorpus.map(sample => sample.output),
      "reasoning</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris",
      "<think>\u4E2D\u{1F600}</think>a\uFEFF<tool_call>invalid</tool_call>\uFFFD",
      "<think></think><think>again</think>",
    ];
    for (const output of samples) {
      for (const prefix of ["", "The answer is", "<think>pending", "<tool_call>get_weather"]) {
        const fast = new GlmParser(tokenizer, { continue_final_message: true });
        const old = new GlmParser(fallback, { continue_final_message: true });
        fast.continueFrom(encode(prefix));
        old.continueFrom(encode(prefix));
        for (const id of encode(output)) {
          assert.deepEqual(fast.onToken(id), old.onToken(id));
          assert.equal(fast.state, old.state);
          assert.equal(fast.producedToolCalls, old.producedToolCalls);
        }
        assert.deepEqual(fast.finish(), old.finish());
        assert.deepEqual(fast.finish(), old.finish());
      }
    }
  });

  it("bounds decode input work on long ASCII output", (t) => {
    const lengths: number[] = [];
    const measured = new Proxy(tokenizer, {
      get(target, key, receiver) {
        if (key === "decode") return (ids: number[], options: Parameters<Tokenizer["decode"]>[1]) => {
          lengths.push(ids.length);
          return target.decode(ids, options);
        };
        return Reflect.get(target, key, receiver);
      },
    });
    const count = 4096;
    const parser = new DefaultChatModelParser(measured);
    for (let i = 0; i < count; i++) assert.deepEqual(parser.onToken(byteIds[0x61]), events("a"));
    assert.deepEqual(parser.finish(), []);
    assert.ok(Math.max(...lengths) <= 2);
    const work = lengths.reduce((sum, length) => sum + length, 0);
    assert.ok(work <= 3 * count);
    t.diagnostic(`decode token inputs: fast=${work}, full-prefix=${count * (count + 1) / 2 + count}`);
  });
});

describe("incremental decoder feature guard", () => {
  for (const [name, internalSuffix, modelSuffix] of [
    ["internal suffix", "</w>", undefined],
    ["JSON model suffix fallback", undefined, "</w>"],
    ["internal suffix overrides empty model suffix", "</w>", ""],
  ] as const) {
    it(`retains full history with ${name}`, () => {
      const calls: number[][] = [];
      const tokenizer = {
        _tokenizerJSON: { decoder: { type: "ByteLevel" }, model: { end_of_word_suffix: modelSuffix } },
        _tokenizerConfig: { clean_up_tokenization_spaces: false },
        _tokenizer: { decoder: { end_of_word_suffix: internalSuffix } },
        decode(ids: number[]) {
          calls.push([...ids]);
          return ids.map(id => String.fromCharCode(id)).join("");
        },
      } as unknown as Tokenizer;
      const parser = new DefaultChatModelParser(tokenizer);
      for (const id of [65, 66, 67]) parser.onToken(id);
      parser.finish();
      assert.deepEqual(calls, [[65], [65, 66], [65, 66, 67], [65, 66, 67]]);
    });
  }

  it("retains anchors with BOM-stripping TextDecoder and literal added-token boundaries", () => {
    const decoder = new TextDecoder("utf-8");
    const literals = new Map([[256, "<literal>"], [257, "\uFFFD"], [258, "\uFEFF"]]);
    const tokenizer = {
      _tokenizerJSON: { decoder: { type: "ByteLevel" } },
      _tokenizerConfig: { clean_up_tokenization_spaces: false },
      decode(ids: number[]) {
        let text = "";
        let bytes: number[] = [];
        for (const id of ids) {
          const literal = literals.get(id);
          if (literal !== undefined) {
            text += decoder.decode(Uint8Array.from(bytes)) + literal;
            bytes = [];
          } else bytes.push(id);
        }
        return text + decoder.decode(Uint8Array.from(bytes));
      },
    } as unknown as Tokenizer;
    for (const ids of [
      [...Buffer.from("a\uFEFFb\uFEFF")],
      [...Buffer.from("\uFEFF\uFEFFa\uFEFF")],
      [97, 0xE4, 0xB8, 0xAD, 0xEF, 0xBB, 0xBF, 98],
      [97, 256, 0xEF, 0xBB, 0xBF, 98, 257, 258, 0xEF, 0xBB, 0xBF, 99],
      [97, 0xE4, 257, 0xB8, 0xAD, 256, 0xEF, 0xBB, 0xBF, 98],
    ]) {
      // Flush at every truncation point, including incomplete UTF-8 and BOMs.
      for (let end = 0; end <= ids.length; end++) {
        const fast = new DefaultChatModelParser(tokenizer);
        const old = new FullPrefixDecoder(tokenizer);
        for (const id of ids.slice(0, end)) assert.deepEqual(fast.onToken(id), events(old.push(id)));
        assert.deepEqual(fast.finish(), events(old.flush()));
      }
    }
  });

  for (const [name, type, config, internal, bounded] of [
    ["cleanup enabled", "ByteLevel", true, true, false],
    ["non ByteLevel", "WordPiece", false, false, false],
    ["missing metadata", undefined, false, false, false],
    ["unknown cleanup", "ByteLevel", undefined, undefined, false],
    ["non-boolean cleanup", "ByteLevel", 0, 0, false],
    ["internal true overrides config false", "ByteLevel", false, true, false],
    ["internal false overrides config true", "ByteLevel", true, false, true],
    ["config false default", "ByteLevel", false, undefined, true],
  ] as const) {
    it(name, () => {
      const calls: number[][] = [];
      const tokenizer = {
        _tokenizerJSON: { decoder: { type } },
        _tokenizerConfig: { clean_up_tokenization_spaces: config },
        _tokenizer: { clean_up_tokenization_spaces: internal },
        decode(ids: number[], options: unknown) {
          assert.deepEqual(options, { skip_special_tokens: false });
          calls.push([...ids]);
          return ids.map(id => String.fromCharCode(id)).join("");
        },
      } as unknown as Tokenizer;
      const parser = new DefaultChatModelParser(tokenizer);
      for (const id of [65, 66, 67, 68]) assert.deepEqual(parser.onToken(id), events(String.fromCharCode(id)));
      assert.deepEqual(parser.finish(), []);
      if (bounded) assert.ok(calls.every(ids => ids.length <= 2));
      else assert.deepEqual(calls, [[65], [65, 66], [65, 66, 67], [65, 66, 67, 68], [65, 66, 67, 68]]);
    });
  }
});
