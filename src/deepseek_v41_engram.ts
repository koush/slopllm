import { createHash } from "node:crypto";
import type { Tokenizer } from "./chat_model";

/** Tokenizer-derived constants uploaded for the GPU Engram hash operator. */
export class DeepseekV41EngramHasher {
  readonly tokenMap: Int32Array;
  readonly primes: number[][] = [];
  readonly offsets: number[][] = [];

  constructor(tokenizer: Tokenizer) {
    const vocab = tokenizer.get_vocab();
    const raw = new Map([...vocab.entries()].map(([token, id]) => [Number(id), token]));
    this.tokenMap = new Int32Array(raw.size);
    const keys = new Map<string, number>();
    for (let id = 0; id < this.tokenMap.length; id++) {
      const text = tokenizer.decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false });
      let key = raw.get(id)!;
      if (!text.includes("\ufffd")) {
        key = text.normalize("NFKC").normalize("NFD").replace(/\p{M}/gu, "").toLowerCase()
          .replace(/[ \t\r\n]+/g, " ").replace(/^ $/, "\ue000")
          .replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "").replace(/\ue000/g, " ") || text;
      }
      if (!keys.has(key)) keys.set(key, keys.size);
      this.tokenMap[id] = keys.get(key)!;
    }
    const hash = createHash("sha256").update(Buffer.from(this.tokenMap.buffer)).digest("hex");
    if (keys.size !== 99092 || hash !== "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481") {
      throw new Error("DeepSeek V4.1 Engram token normalization does not match the checkpoint tokenizer");
    }
    const seen = new Set<number>();
    const prime = (n: number) => {
      if (n % 2 === 0) return false;
      for (let d = 3; d * d <= n; d += 2) if (n % d === 0) return false;
      return true;
    };
    for (let layer = 0; layer < 2; layer++) {
      const primes: number[] = [], offsets: number[] = [];
      let offset = 0;
      for (let order = 0; order < 3; order++) {
        let candidate = 15999999;
        for (let head = 0; head < 8; head++) {
          do { candidate++; } while (!prime(candidate) || seen.has(candidate));
          seen.add(candidate);
          primes.push(candidate);
          offsets.push(offset);
          offset += candidate;
        }
      }
      if (offset !== [384006168, 384016682][layer]) throw new Error("DeepSeek V4.1 Engram table geometry mismatch");
      this.primes.push(primes);
      this.offsets.push(offsets);
    }
  }

}
