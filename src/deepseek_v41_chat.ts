import { ChatModelParser, resolveControlToken } from "./chat-model-parser";
import type { ChatTemplateKwargs, Tokenizer } from "./chat_model";

/** Text-only V4.1 prompt protocol; the checkpoint ships no Jinja template. */
export const DEEPSEEK_V41_CHAT_TEMPLATE = `
 {%- if tools -%}{{ raise_exception('V4.1 chat does not support tools') }}{%- endif -%}
{%- set enabled = true -%}
{%- if enable_thinking is defined or thinking is defined -%}
  {%- set enabled = (enable_thinking | default(false)) or (thinking | default(false)) -%}
{%- endif -%}
{%- set effort = reasoning_effort | default('high') -%}
{%- if effort is none -%}{%- set effort = 'high' -%}{%- endif -%}
{%- if effort == 'none' -%}{%- set enabled = false -%}{%- set effort = 'high' -%}{%- endif -%}
{%- set budgets = {'low': 50, 'high': 75, 'xhigh': 75, 'max': 100} -%}
{%- if effort is string and effort in budgets -%}{%- set budget = budgets[effort] -%}
{%- elif effort is number and effort >= 1 and effort <= 100 and effort == (effort | int) -%}{%- set budget = effort -%}
{%- else -%}{{ raise_exception('Invalid V4.1 reasoning_effort') }}{%- endif -%}
{%- set last = namespace(user=-1) -%}
{%- for message in messages -%}
  {%- if message['role'] == 'user' or (message['role'] == 'system' and not loop.first) -%}{%- set last.user = loop.index0 -%}{%- endif -%}
{%- endfor -%}
{{- bos_token -}}
{%- if enabled -%}{{ '<｜System｜>Reasoning Effort: ' + (budget | string) + ' (range 1-100, the higher the value, the more thorough the reasoning)\n\n' }}{%- endif -%}
{%- for message in messages -%}
  {%- set role = message['role'] -%}
  {%- if message['tool_calls'] or message['task'] -%}{{ raise_exception('V4.1 chat supports ordinary text messages only') }}{%- endif -%}
  {%- if role == 'system' -%}
    {%- if not loop.first or not enabled -%}{{ '<｜System｜>' }}{%- endif -%}
  {%- elif role == 'user' -%}
    {%- if not loop.first and messages[loop.index0 - 1]['role'] == 'user' -%}{{ '\n\n' }}{%- else -%}{{ '<｜User｜>' }}{%- endif -%}
  {%- elif role == 'assistant' -%}
    {%- if enabled and (not (drop_thinking | default(true)) or loop.index0 > last.user) -%}{{ (message['reasoning_content'] | default('', true)) + '</think>' }}{%- endif -%}
  {%- else -%}{{ raise_exception('Unsupported V4.1 text message role: ' + role) }}{%- endif -%}
  {%- if message['content'] is string -%}{{ message['content'] }}
  {%- elif message['content'] -%}
    {%- for part in message['content'] -%}
      {%- if part['type'] != 'text' -%}{{ raise_exception('V4.1 chat supports text content only') }}{%- endif -%}
      {%- if not loop.first -%}{{ '\n\n' }}{%- endif -%}{{ part['text'] }}
    {%- endfor -%}
  {%- endif -%}
  {%- if role == 'assistant' and not (loop.last and (continue_final_message | default(false))) -%}{{ eos_token }}{%- endif -%}
  {%- if (role == 'user' or (role == 'system' and not loop.first)) and ((loop.last and add_generation_prompt) or (not loop.last and messages[loop.index0 + 1]['role'] == 'assistant')) -%}
    {{- '<｜Assistant｜>' -}}
    {%- if enabled and (not (drop_thinking | default(true)) or loop.index0 >= last.user) -%}{{ '<think>' }}{%- else -%}{{ '</think>' }}{%- endif -%}
  {%- endif -%}
{%- endfor -%}`;

export function dsv41Thinking(kwargs: ChatTemplateKwargs): boolean {
  if (kwargs.reasoning_effort === "none") return false;
  if (kwargs.thinking === undefined && kwargs.enable_thinking === undefined) return true;
  return !!(kwargs.thinking || kwargs.enable_thinking);
}

export class DeepseekV41ChatParser extends ChatModelParser {
  constructor(tokenizer: Tokenizer, kwargs: ChatTemplateKwargs) {
    super(tokenizer, {
      thinkStart: resolveControlToken(tokenizer, "<think>"),
      thinkEnd: resolveControlToken(tokenizer, "</think>"),
      toolCallStart: -1, toolCallEnd: -2,
    }, dsv41Thinking(kwargs) ? "reasoning" : "content");
  }

  protected parseToolCall(): never { throw new Error("V4.1 chat does not support tools"); }
}
