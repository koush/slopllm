export interface GlmParserCorpusEntry {
  name: string;
  output: string;
  reasoning: string;
  content: string;
  toolCalls: Array<{ name: string; arguments: Record<string, unknown> }>;
}

// Captured from lukealonso/GLM-5.2-NVFP4 with temperature=0. The opening
// <think> is part of the rendered prompt, so these strings begin directly with
// the first generated reasoning token.
export const glmParserCorpus: GlmParserCorpusEntry[] = [
  {
    name: "plain response",
    output: `1.  **Analyze the Request:**
    *   Input: "Reply with exactly the word READY and nothing else."
    *   Constraint 1: Reply with exactly the word "READY".
    *   Constraint 2: Nothing else (no punctuation, no extra words, no spaces if possible, just the word).

2.  **Formulate the Output:**
    *   Output: READY

3.  **Verify Constraints:**
    *   Is it exactly the word READY? Yes.
    *   Is there anything else? No.

4.  **Finalize Output:** READY</think>READY`,
    reasoning: `1.  **Analyze the Request:**
    *   Input: "Reply with exactly the word READY and nothing else."
    *   Constraint 1: Reply with exactly the word "READY".
    *   Constraint 2: Nothing else (no punctuation, no extra words, no spaces if possible, just the word).

2.  **Formulate the Output:**
    *   Output: READY

3.  **Verify Constraints:**
    *   Is it exactly the word READY? Yes.
    *   Is there anything else? No.

4.  **Finalize Output:** READY`,
    content: "READY",
    toolCalls: [],
  },
  {
    name: "content followed by a string argument tool call",
    output: `The user wants to know the current weather in Paris. I'll call the get_weather function with "Paris" as the city parameter.</think>I'll fetch the current weather in Paris for you right away.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`,
    reasoning: `The user wants to know the current weather in Paris. I'll call the get_weather function with "Paris" as the city parameter.`,
    content: "I'll fetch the current weather in Paris for you right away.",
    toolCalls: [{ name: "get_weather", arguments: { city: "Paris" } }],
  },
  {
    name: "mixed scalar and array arguments",
    output: `The user wants to set an alarm with the following details:
- Name: workout
- Hour: 6
- Minute: 30
- Weekdays: Monday through Friday (weekday alarm)
- Volume: 8
- Vibrate: true</think>I'll set up your weekday workout alarm right away!<tool_call>set_alarm<arg_key>name</arg_key><arg_value>workout</arg_value><arg_key>hour</arg_key><arg_value>6</arg_value><arg_key>minute</arg_key><arg_value>30</arg_value><arg_key>weekdays</arg_key><arg_value>["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]</arg_value><arg_key>volume</arg_key><arg_value>8</arg_value><arg_key>vibrate</arg_key><arg_value>true</arg_value></tool_call>`,
    reasoning: `The user wants to set an alarm with the following details:
- Name: workout
- Hour: 6
- Minute: 30
- Weekdays: Monday through Friday (weekday alarm)
- Volume: 8
- Vibrate: true`,
    content: "I'll set up your weekday workout alarm right away!",
    toolCalls: [{
      name: "set_alarm",
      arguments: {
        name: "workout",
        hour: 6,
        minute: 30,
        weekdays: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        volume: 8,
        vibrate: true,
      },
    }],
  },
  {
    name: "array arguments without visible content",
    output: `The user wants me to search for articles about TypeScript parsers tagged with "streaming" or "tokenizer", limited to 5 results, sorted by newest first.</think><tool_call>search_articles<arg_key>query</arg_key><arg_value>TypeScript parser</arg_value><arg_key>tags</arg_key><arg_value>["streaming", "tokenizer"]</arg_value><arg_key>limit</arg_key><arg_value>5</arg_value><arg_key>sort</arg_key><arg_value>newest</arg_value></tool_call>`,
    reasoning: `The user wants me to search for articles about TypeScript parsers tagged with "streaming" or "tokenizer", limited to 5 results, sorted by newest first.`,
    content: "",
    toolCalls: [{
      name: "search_articles",
      arguments: {
        query: "TypeScript parser",
        tags: ["streaming", "tokenizer"],
        limit: 5,
        sort: "newest",
      },
    }],
  },
  {
    name: "adjacent tool calls",
    output: `The user wants to compare the current weather in Paris and Tokyo. I need to call get_weather for both cities. Since these calls are independent, I can make them simultaneously.</think>I'll fetch the current weather for both Paris and Tokyo simultaneously since the calls are independent of each other.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>`,
    reasoning: "The user wants to compare the current weather in Paris and Tokyo. I need to call get_weather for both cities. Since these calls are independent, I can make them simultaneously.",
    content: "I'll fetch the current weather for both Paris and Tokyo simultaneously since the calls are independent of each other.",
    toolCalls: [
      { name: "get_weather", arguments: { city: "Paris" } },
      { name: "get_weather", arguments: { city: "Tokyo" } },
    ],
  },
  {
    name: "tool without arguments",
    output: "The user wants me to call the get_current_time function, which takes no arguments.</think><tool_call>get_current_time</tool_call>",
    reasoning: "The user wants me to call the get_current_time function, which takes no arguments.",
    content: "",
    toolCalls: [{ name: "get_current_time", arguments: {} }],
  },
  {
    name: "nested object argument",
    output: `The user wants me to submit a job with the following details:
- name: "nightly-build"
- metadata: branch "main", retries 2

I have all the required parameters, so I can make the call.</think>I'll submit the job "nightly-build" with the specified metadata right away.<tool_call>submit_job<arg_key>name</arg_key><arg_value>nightly-build</arg_value><arg_key>metadata</arg_key><arg_value>{"branch": "main", "retries": 2}</arg_value></tool_call>`,
    reasoning: `The user wants me to submit a job with the following details:
- name: "nightly-build"
- metadata: branch "main", retries 2

I have all the required parameters, so I can make the call.`,
    content: `I'll submit the job "nightly-build" with the specified metadata right away.`,
    toolCalls: [{
      name: "submit_job",
      arguments: {
        name: "nightly-build",
        metadata: { branch: "main", retries: 2 },
      },
    }],
  },
];
