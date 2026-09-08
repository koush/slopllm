import { type Tokenizer } from "./chat_model";

export type OutputParserState = "reasoning" | "content" | "tool";

export type OutputParserEvent =
  | { type: "reasoning_delta"; text: string }
  | { type: "content_delta"; text: string }
  | { type: "tool_call"; index: number; name: string; arguments: string }
  | { type: "parse_error"; message: string; raw: string };

export interface OutputControlTokens {
  thinkStart: number;
  thinkEnd: number;
  toolCallStart: number;
  toolCallEnd: number;
}

export interface ParsedToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * Stateful parser for one generated assistant response.
 *
 * Structural markers are handled by token ID. Ordinary text is decoded from
 * a retained token prefix so byte-level tokenizer fragments are not emitted
 * prematurely (unsupported decoders retain the complete prefix).
 */
export abstract class ChatModelParser {
  private readonly textDecoder: IncrementalTokenDecoder;
  private toolTokenIds: number[] = [];
  private toolCallIndex = 0;
  private _producedToolCalls = false;

  protected constructor(
    protected readonly tokenizer: Tokenizer,
    protected readonly controlTokens: OutputControlTokens | undefined,
    private _state: OutputParserState,
  ) {
    this.textDecoder = new IncrementalTokenDecoder(tokenizer);
  }

  get state(): OutputParserState {
    return this._state;
  }

  continueFrom(tokenIds: readonly number[]): void {
    for (const tokenId of tokenIds) this.onToken(tokenId);
    this.flushText();
  }

  get producedToolCalls(): boolean {
    return this._producedToolCalls;
  }

  onToken(tokenId: number): OutputParserEvent[] {
    const controlTokens = this.controlTokens;
    if (controlTokens === undefined) {
      const text = this.textDecoder.push(tokenId);
      return text ? [{ type: "content_delta", text }] : [];
    }

    if (this._state === "tool") {
      if (tokenId !== controlTokens.toolCallEnd) {
        this.toolTokenIds.push(tokenId);
        return [];
      }

      const raw = this.decode(this.toolTokenIds);
      this.toolTokenIds = [];
      this._state = "content";

      try {
        const call = this.parseToolCall(raw);
        const event: OutputParserEvent = {
          type: "tool_call",
          index: this.toolCallIndex++,
          name: call.name,
          arguments: JSON.stringify(call.arguments),
        };
        this._producedToolCalls = true;
        return [event];
      } catch (error) {
        return [{
          type: "parse_error",
          message: error instanceof Error ? error.message : String(error),
          raw,
        }];
      }
    }

    if (tokenId === controlTokens.thinkStart) {
      const events = this.flushText();
      this._state = "reasoning";
      return events;
    }

    if (tokenId === controlTokens.thinkEnd) {
      const events = this.flushText();
      this._state = "content";
      return events;
    }

    if (tokenId === controlTokens.toolCallStart) {
      const events = this.flushText();
      this._state = "tool";
      this.toolTokenIds = [];
      return events;
    }

    const text = this.textDecoder.push(tokenId);
    return text ? [this.textEvent(text)] : [];
  }

  finish(): OutputParserEvent[] {
    if (this._state === "tool") {
      const raw = this.decode(this.toolTokenIds);
      this.toolTokenIds = [];
      this._state = "content";
      return [{
        type: "parse_error",
        message: "Generation ended inside a tool call",
        raw,
      }];
    }

    return this.flushText();
  }

  protected abstract parseToolCall(raw: string): ParsedToolCall;

  private flushText(): OutputParserEvent[] {
    const text = this.textDecoder.flush();
    return text ? [this.textEvent(text)] : [];
  }

  private textEvent(text: string): OutputParserEvent {
    return this._state === "reasoning"
      ? { type: "reasoning_delta", text }
      : { type: "content_delta", text };
  }

  private decode(tokenIds: number[]): string {
    return this.tokenizer.decode(tokenIds, { skip_special_tokens: false });
  }
}

/** Fallback parser for models without a structured output protocol. */
export class DefaultChatModelParser extends ChatModelParser {
  constructor(tokenizer: Tokenizer) {
    super(tokenizer, undefined, "content");
  }

  protected parseToolCall(_raw: string): ParsedToolCall {
    throw new Error("DefaultChatModelParser does not parse tool calls");
  }
}

export function resolveControlToken(tokenizer: Tokenizer, marker: string): number {
  const encoded = tokenizer.encode(marker, { add_special_tokens: false }) as number[];
  if (encoded.length !== 1) {
    throw new Error(`Expected ${JSON.stringify(marker)} to encode to one token, got ${encoded.length}`);
  }
  return encoded[0];
}

class IncrementalTokenDecoder {
  private tokenIds: number[] = [];
  private emittedText = "";
  private readonly canCompactHistory: boolean;

  constructor(private readonly tokenizer: Tokenizer) {
    const internals = tokenizer as unknown as {
      _tokenizerJSON?: { decoder?: { type?: string }; model?: { end_of_word_suffix?: unknown } };
      _tokenizerConfig?: { clean_up_tokenization_spaces?: unknown };
      _tokenizer?: { clean_up_tokenization_spaces?: unknown; decoder?: { end_of_word_suffix?: unknown } };
    };
    // decode_single delegates to the internal tokenizer's cleanup default.
    const cleanup = internals._tokenizer?.clean_up_tokenization_spaces
      ?? internals._tokenizerConfig?.clean_up_tokenization_spaces ?? true;
    // Suffix replacement runs after decoding, independently of cleanup, and
    // can rewrite text spanning multiple otherwise standalone tokens.
    const suffix = internals._tokenizer?.decoder?.end_of_word_suffix
      ?? internals._tokenizerJSON?.model?.end_of_word_suffix;
    this.canCompactHistory = internals._tokenizerJSON?.decoder?.type === "ByteLevel"
      && cleanup === false && !suffix;
  }

  push(tokenId: number): string {
    this.tokenIds.push(tokenId);
    const text = this.tokenizer.decode(this.tokenIds, { skip_special_tokens: false });

    let safeEnd = text.length;
    while (safeEnd > 0 && text.charCodeAt(safeEnd - 1) === 0xFFFD) safeEnd--;

    const safeText = text.slice(0, safeEnd);
    const delta = this.deltaFrom(safeText);
    this.emittedText = safeText;
    if (this.canCompactHistory && safeEnd === text.length) {
      const lastText = this.tokenIds.length === 1
        ? text
        : this.tokenizer.decode([tokenId], { skip_special_tokens: false });
      if (lastText && !lastText.includes("\uFFFD") && safeText.endsWith(lastText)) {
        // Keep a real anchor, not an empty history: restarting byte decoding can
        // change BOM handling. Added tokens also preserve decoder boundaries.
        // Fragment-only streams can retain unbounded history until a standalone
        // token arrives; retain the previous anchor and entire pending group.
        this.tokenIds = [tokenId];
        this.emittedText = lastText;
      }
    }
    return delta;
  }

  flush(): string {
    if (this.tokenIds.length === 0) return "";

    const text = this.tokenizer.decode(this.tokenIds, { skip_special_tokens: false });
    const delta = this.deltaFrom(text);
    this.tokenIds = [];
    this.emittedText = "";
    return delta;
  }

  private deltaFrom(text: string): string {
    if (text.startsWith(this.emittedText)) return text.slice(this.emittedText.length);

    let commonPrefix = 0;
    while (
      commonPrefix < this.emittedText.length &&
      commonPrefix < text.length &&
      this.emittedText[commonPrefix] === text[commonPrefix]
    ) {
      commonPrefix++;
    }
    return text.slice(commonPrefix);
  }
}
