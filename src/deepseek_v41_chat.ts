import { ChatModelParser, IncrementalTokenDecoder, resolveControlToken, type OutputParserEvent, type ParsedToolCall } from "./chat-model-parser";
import type { ChatTemplateKwargs, Tokenizer } from "./chat_model";

/** V4.1 text and spaced-DSML tool protocol; the checkpoint ships no Jinja template. */
export const DEEPSEEK_V41_CHAT_TEMPLATE = `
{%- macro text(content) -%}
  {%- if content is string -%}{{ content }}
  {%- elif content -%}
    {%- for part in content -%}
      {%- if part['type'] != 'text' -%}{{ raise_exception('V4.1 chat supports text content only') }}{%- endif -%}
      {%- if not loop.first -%}{{ '\n\n' }}{%- endif -%}{{ part['text'] }}
    {%- endfor -%}
  {%- endif -%}
{%- endmacro -%}
{%- macro tool_prompt(tools) -%}
{{- '## Tools\n\nYou have access to a set of tools to help answer the user\\'s question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="$TOOL_NAME">\n<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="$TOOL_NAME2">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set \`string="true"\`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set \`string="false"\`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n' -}}
{%- for tool in tools -%}
  {%- if tool['type'] != 'function' or not tool['function'] -%}{{ raise_exception('V4.1 requires function tools') }}{%- endif -%}
  {%- if not loop.first -%}{{ '\n' }}{%- endif -%}{{ tool['function'] | tojson }}
{%- endfor -%}
{{- '\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n' -}}
{%- endmacro -%}
{# Tool results have no standalone role in V4.1. Restore call order before
   converting contiguous tool replies into user content. #}
{%- set normalized = namespace(messages=[], calls=[]) -%}
{%- for message in messages -%}
  {%- if message['role'] == 'assistant' -%}{%- set normalized.calls = message['tool_calls'] | default([], true) -%}{%- endif -%}
  {%- if message['role'] != 'tool' -%}
    {%- set normalized.messages = normalized.messages + [message] -%}
  {%- elif loop.first or messages[loop.index0 - 1]['role'] != 'tool' -%}
    {%- set group = namespace(results=[], done=false, ordered=[]) -%}
    {%- for reply in messages[loop.index0:] -%}
      {%- if reply['role'] != 'tool' -%}{%- set group.done = true -%}{%- endif -%}
      {%- if not group.done -%}{%- set group.results = group.results + [reply] -%}{%- endif -%}
    {%- endfor -%}
    {%- for call in normalized.calls -%}
      {%- for reply in group.results -%}
        {%- if call['id'] and reply['tool_call_id'] == call['id'] -%}{%- set group.ordered = group.ordered + [reply] -%}{%- endif -%}
      {%- endfor -%}
    {%- endfor -%}
    {%- for reply in group.results -%}
      {%- if reply not in group.ordered -%}{%- set group.ordered = group.ordered + [reply] -%}{%- endif -%}
    {%- endfor -%}
    {%- for reply in group.ordered -%}
      {%- set normalized.messages = normalized.messages + [{'role': 'user', 'content': '<tool_result>' + text(reply['content']) + '</tool_result>'}] -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{%- set messages = normalized.messages -%}
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
{%- if tools and (not messages or messages[0]['role'] != 'system') -%}
  {%- if not enabled -%}{{ '<｜System｜>' }}{%- endif -%}
  {{- '\n\n' + tool_prompt(tools) -}}
{%- endif -%}
{%- for message in messages -%}
  {%- set role = message['role'] -%}
  {%- if message['task'] -%}{{ raise_exception('V4.1 chat does not support task messages') }}{%- endif -%}
  {%- if role == 'system' -%}
    {%- if not loop.first or not enabled -%}{{ '<｜System｜>' }}{%- endif -%}
  {%- elif role == 'user' -%}
    {%- if not loop.first and messages[loop.index0 - 1]['role'] == 'user' -%}{{ '\n\n' }}{%- else -%}{{ '<｜User｜>' }}{%- endif -%}
  {%- elif role == 'assistant' -%}
    {%- if enabled and (not (drop_thinking | default(true)) or loop.index0 > last.user) -%}{{ (message['reasoning_content'] | default('', true)) + '</think>' }}{%- endif -%}
  {%- else -%}{{ raise_exception('Unsupported V4.1 text message role: ' + role) }}{%- endif -%}
  {{- text(message['content']) -}}
  {%- if role == 'system' and loop.first and tools -%}{{ '\n\n' + tool_prompt(tools) }}{%- endif -%}
  {%- if role == 'assistant' and message['tool_calls'] -%}
    {{- '\n\n<｜DSML｜ calls>\n' -}}
    {%- for call in message['tool_calls'] -%}
      {%- if not loop.first -%}{{ '\n' }}{%- endif -%}
      {{- '<｜DSML｜ invoke name="' + call['function']['name'] + '">\n' -}}
      {%- set arguments = call['function']['arguments'] -%}
      {%- if arguments is not mapping -%}{{ raise_exception('Tool-call arguments must be normalized to an object') }}{%- endif -%}
      {%- for key, value in arguments.items() -%}
        {%- if not loop.first -%}{{ '\n' }}{%- endif -%}
        {{- '<｜DSML｜ parameter name="' + key + '" string="' -}}
        {%- if value is string -%}{{ 'true">' + value }}{%- else -%}{{ 'false">' + (value | tojson) }}{%- endif -%}
        {{- '</｜DSML｜ parameter>' -}}
      {%- endfor -%}
      {{- '\n</｜DSML｜ invoke>' -}}
    {%- endfor -%}
    {{- '\n</｜DSML｜ calls>' -}}
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
  private readonly toolDecoder: IncrementalTokenDecoder;
  private pending = "";
  private inCalls = false;
  private callIndex = 0;

  constructor(tokenizer: Tokenizer, kwargs: ChatTemplateKwargs) {
    super(tokenizer, {
      thinkStart: resolveControlToken(tokenizer, "<think>"),
      thinkEnd: resolveControlToken(tokenizer, "</think>"),
      toolCallStart: -1, toolCallEnd: -2,
    }, !kwargs.continue_final_message && dsv41Thinking(kwargs) ? "reasoning" : "content");
    this.toolDecoder = new IncrementalTokenDecoder(tokenizer);
  }

  override get state() { return this.inCalls ? "tool" as const : super.state; }
  override get producedToolCalls() { return this.callIndex > 0; }

  override onToken(tokenId: number): OutputParserEvent[] {
    if (this.inCalls) {
      const events = this.consume(this.toolDecoder.push(tokenId));
      if (!this.inCalls) events.push(...this.consume(this.toolDecoder.flush()));
      return events;
    }
    const events = this.consumeEvents(super.onToken(tokenId));
    if (this.inCalls) events.push(...this.consumeEvents(super.finish()));
    return events;
  }

  override finish(): OutputParserEvent[] {
    const events = this.consumeEvents(super.finish());
    events.push(...this.consume(this.toolDecoder.flush()));
    if (this.inCalls) {
      events.push({ type: "parse_error", message: "Generation ended inside a DSML tool-call block", raw: this.pending });
    } else if (this.pending) {
      events.push({ type: "content_delta", text: this.pending });
    }
    this.pending = "";
    this.inCalls = false;
    return events;
  }

  private consumeEvents(events: OutputParserEvent[]): OutputParserEvent[] {
    return events.flatMap(event => event.type === "content_delta" ? this.consume(event.text) : [event]);
  }

  private consume(text: string): OutputParserEvent[] {
    this.pending += text;
    const events: OutputParserEvent[] = [];
    const start = "<｜DSML｜ calls>", end = "</｜DSML｜ calls>", invokeEnd = "</｜DSML｜ invoke>";
    for (;;) {
      if (!this.inCalls) {
        const index = this.pending.indexOf(start);
        if (index < 0) {
          // Markers span multiple tokenizer tokens; retain only a possible prefix.
          let keep = Math.min(start.length - 1, this.pending.length);
          while (keep && !start.startsWith(this.pending.slice(-keep))) keep--;
          const content = this.pending.slice(0, this.pending.length - keep);
          if (content) events.push({ type: "content_delta", text: content });
          this.pending = this.pending.slice(this.pending.length - keep);
          return events;
        }
        if (index) events.push({ type: "content_delta", text: this.pending.slice(0, index) });
        this.pending = this.pending.slice(index + start.length);
        this.inCalls = true;
      }
      const close = this.pending.indexOf(end), invoke = this.pending.indexOf(invokeEnd);
      if (close >= 0 && (invoke < 0 || close < invoke)) {
        const raw = this.pending.slice(0, close);
        if (raw.trim()) events.push({ type: "parse_error", message: "Malformed DSML tool-call block", raw });
        this.pending = this.pending.slice(close + end.length);
        this.inCalls = false;
        continue;
      }
      if (invoke < 0) return events;
      const raw = this.pending.slice(0, invoke + invokeEnd.length);
      this.pending = this.pending.slice(invoke + invokeEnd.length);
      try {
        const call = this.parseToolCall(raw);
        events.push({ type: "tool_call", index: this.callIndex++, name: call.name, arguments: JSON.stringify(call.arguments) });
      } catch (error) {
        events.push({ type: "parse_error", message: error instanceof Error ? error.message : String(error), raw });
      }
    }
  }

  protected parseToolCall(raw: string): ParsedToolCall {
    const invoke = /^\s*<｜DSML｜ invoke name="([^"]+)">([\s\S]*)<\/｜DSML｜ invoke>$/.exec(raw);
    if (!invoke || !invoke[1].trim()) throw new Error("Malformed DSML invocation");
    const args: Record<string, unknown> = Object.create(null);
    const parameters = invoke[2];
    const parameter = /\s*<｜DSML｜ parameter name="([^"]+)" string="(true|false)">([\s\S]*?)<\/｜DSML｜ parameter>/y;
    let offset = 0;
    while (parameters.slice(offset).trim()) {
      parameter.lastIndex = offset;
      const match = parameter.exec(parameters);
      if (!match || !match[1].trim() || Object.hasOwn(args, match[1])) throw new Error("Malformed or duplicate DSML parameter");
      args[match[1]] = match[2] === "true" ? match[3] : JSON.parse(match[3]);
      offset = parameter.lastIndex;
    }
    return { name: invoke[1], arguments: args };
  }
}
