import { type Tokenizer } from "./chat_model";
import {
  type OutputControlTokens,
  type ParsedToolCall,
  ChatModelParser,
  resolveControlToken,
} from "./chat-model-parser";

export interface GlmChatTemplateKwargs {
  continue_final_message?: boolean;
  enable_thinking?: boolean;
  reasoning_effort?: "low" | "high" | "max";
}

const templateStates = new WeakMap<Tokenizer, { template: string; options: string; reasoning: boolean }>();

function startsInReasoning(tokenizer: Tokenizer, kwargs: GlmChatTemplateKwargs): boolean {
  // Continuations establish their state by consuming the existing assistant
  // text through continueFrom(), not by adding a new generation prompt.
  if (kwargs.continue_final_message) return false;

  const template = tokenizer.get_chat_template();
  const options = JSON.stringify(kwargs);
  const cached = templateStates.get(tokenizer);
  if (cached?.template === template && cached.options === options) return cached.reasoning;

  // GLM-5.1/5.2 can close the thinking block when enable_thinking is false;
  // GLM-5.3 ignores that option and always opens it. Probe rendered behavior
  // rather than model architecture or the presence of a variable in Jinja.
  const rendered = tokenizer.apply_chat_template([{ role: "user", content: "Hello" }], {
    ...kwargs,
    tokenize: false,
    add_generation_prompt: true,
  });
  const assistantPrefix = rendered.slice(rendered.lastIndexOf("<|assistant|>") + "<|assistant|>".length);
  const reasoning = assistantPrefix.lastIndexOf("<think>") > assistantPrefix.lastIndexOf("</think>");
  templateStates.set(tokenizer, { template, options, reasoning });
  return reasoning;
}

/** Parser for the GLM chat-template output protocol. */
export class GlmParser extends ChatModelParser {
  constructor(tokenizer: Tokenizer, kwargs: GlmChatTemplateKwargs = {}) {
    // GLM pre-fills <think> in the generation prompt. The model therefore
    // starts by generating reasoning text rather than the opening marker.
    super(
      tokenizer,
      resolveGlmControlTokens(tokenizer),
      startsInReasoning(tokenizer, kwargs) ? "reasoning" : "content",
    );
  }

  protected parseToolCall(raw: string): ParsedToolCall {
    const firstArgument = raw.indexOf("<arg_key>");
    const name = (firstArgument === -1 ? raw : raw.slice(0, firstArgument)).trim();
    if (!name) throw new Error("GLM tool call has no function name");

    const args: Record<string, unknown> = {};
    const argumentPattern = /<arg_key>([\s\S]*?)<\/arg_key><arg_value>([\s\S]*?)<\/arg_value>/g;
    let match: RegExpExecArray | null;
    let argumentCount = 0;

    while ((match = argumentPattern.exec(raw)) !== null) {
      const key = match[1].trim();
      if (!key) throw new Error("GLM tool call contains an empty argument name");
      args[key] = parseArgumentValue(match[2]);
      argumentCount++;
    }

    if (firstArgument !== -1 && argumentCount === 0) {
      throw new Error("GLM tool call contains malformed arguments");
    }

    return { name, arguments: args };
  }
}

function resolveGlmControlTokens(tokenizer: Tokenizer): OutputControlTokens {
  return {
    thinkStart: resolveControlToken(tokenizer, "<think>"),
    thinkEnd: resolveControlToken(tokenizer, "</think>"),
    toolCallStart: resolveControlToken(tokenizer, "<tool_call>"),
    toolCallEnd: resolveControlToken(tokenizer, "</tool_call>"),
  };
}

function parseArgumentValue(raw: string): unknown {
  const value = raw.trim();
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}
