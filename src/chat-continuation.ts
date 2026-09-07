import type { ChatTemplateKwargs, Tokenizer } from "./chat_model";

export function tokenizeContinuation(
  tokenizer: Tokenizer,
  messages: { role: string; content?: unknown }[],
  tools: unknown[] | undefined,
  kwargs: ChatTemplateKwargs,
): number[] {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant" || typeof last.content !== "string" || !last.content.trim()) {
    throw new Error("continue_final_message requires a final assistant message with non-empty text content");
  }
  if (kwargs.add_generation_prompt === true) {
    throw new Error("continue_final_message cannot be combined with add_generation_prompt");
  }
  const rendered = tokenizer.apply_chat_template(messages as any, {
    ...kwargs, tools, tokenize: false, add_generation_prompt: false,
  } as any) as unknown as string;
  const content = last.content.trim();
  const offset = rendered.lastIndexOf(content);
  if (offset === -1) throw new Error("Final assistant content was not preserved by the chat template");
  // Remove the template's end-of-message markers, but preserve the supplied prefix's trailing whitespace.
  const trailing = last.content.slice(last.content.trimEnd().length);
  return tokenizer.encode(rendered.slice(0, offset + content.length) + trailing, { add_special_tokens: false });
}
