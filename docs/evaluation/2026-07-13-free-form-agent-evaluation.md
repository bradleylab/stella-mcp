# Free-Form Agent Evaluation

## Scope

This evaluation records one free-form agent sample against the versioned
three-scenario protocol in `evaluation/agent_scenarios.json`. It used the
personal OpenAI endpoint, `gpt-5.6-sol`, Chat Completions function calling, a
4096-token completion cap, no temperature or seed parameter, and
`reasoning_effort="none"`. The model received the 42-tool live Stella MCP
catalog through a fresh stdio session for each scenario.

## Endpoint Preflight

The first endpoint attempt was rejected before model inference because GPT-5.6
Sol does not accept Chat Completions function tools at its default reasoning
effort. The preserved preflight report records no resolved model, token usage,
or MCP tool calls. The endpoint's required `reasoning_effort="none"` parameter
was then added to the backend as an explicit run parameter and covered by
offline tests. This was an API compatibility correction, not a repeated model
sample.

## Recorded Result

The valid sample completed all three scenarios in 54.368 seconds. It made 17
successful MCP calls with no tool errors and produced all five expected
artifacts. Two scenarios passed every preregistered criterion. The construction
scenario completed its complete build, validate, simulate, render, and strict
save sequence, but the protocol marked it failed because the model set the time
unit to `Year` while the exact check required `Years`. The task prompt used
`1/Year` in its equations and did not require the plural label, so this is a
known strictness limitation in the version 1 protocol; the recorded status has
not been changed after the run.

Manual review found that the final responses agree with the retained tool
outputs and artifacts. The growth run ended at 161.051 people. The modified SIR
model ended with 208.19 susceptible, 14.33 infected, and 777.48 recovered, with
a maximum infected value of 159.41. The accumulator CSVs record final values of
4 and 8 for baseline and doubled rate, and sensitivity values of 4, 8, and 12
for rates 1, 2, and 3.

## Evidence

- [`0.12.0-agent-evaluation.json`](../../results/evaluation/0.12.0-agent-evaluation.json)
  is the machine-readable record.
- [`0.12.0-agent-evaluation.md`](../../results/evaluation/0.12.0-agent-evaluation.md)
  is its generated human-readable rendering.
- [`0.12.0-agent-preflight-failure.json`](../../results/evaluation/0.12.0-agent-preflight-failure.json)
  and its Markdown rendering preserve the endpoint compatibility rejection.
- `results/evaluation/0.12.0-agent-artifacts/` contains the five output files;
  the report records their byte counts and SHA-256 hashes.

This is one model sample, not an estimate of a success rate. It demonstrates
that the selected agent can complete the template-modification and analysis
workflows and can operationally complete the construction workflow, while also
exposing one brittle protocol criterion.
