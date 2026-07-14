# Free-Form Agent Evaluation Specification

Status: in progress on `codex/agent-evaluation`

## Objective

Measure whether an LLM can plan and complete representative Stella MCP
workflows from user-level requests. Keep this experiment separate from the
deterministic MCP baseline because its behavior depends on the selected model,
endpoint, prompt, sampling controls, and tool-call decisions.

## Fixed Inputs

`evaluation/agent_scenarios.json` is the versioned protocol. It fixes:

- the system prompt shown to the model;
- zero requested temperature, seed `20260713`, and a 4096-token completion cap;
- a 12-round tool-call safety cap;
- three user prompts covering model construction, template modification, and
  analysis of an imported model;
- the required tool order for each requested workflow;
- deterministic post-run MCP checks and expected artifacts.

The endpoint, model, and supported sampling controls are run parameters rather
than scenario inputs. The report must record their identifiers and which
requested sampling controls were sent. The completion cap is always sent.

## Agent Loop

For each scenario, start a fresh real stdio MCP session and discover its tool
catalog. Convert those MCP definitions directly to the endpoint's function-tool
format. Send the fixed system prompt and expanded user prompt, execute every
returned tool call through the live MCP session, and return the complete tool
result to the model. Continue until the model gives a final response or reaches
the configured tool-round cap. Parallel function calling is disabled so each
dependent MCP operation receives the preceding tool result before the model
chooses the next operation.

Paths in prompts are expanded only for execution. Reports replace repository
and artifact-directory paths with stable tokens. API credentials are read from
the selected environment at call time and are never accepted as command-line
values or written to results. The runner refuses to start when a selected
scenario's expected output already exists, preventing stale files from
satisfying artifact checks. The CLI also refuses existing JSON or Markdown
result files so a later invocation cannot silently replace the first sample.

The personal route reads `OPENAI_API_KEY` and pins the official API base rather
than inheriting an ambient institutional URL. The WashU route reads
`OPENAI_BASE_URL`, `WUSTL_TENANT_ID`, `WUSTL_CLIENT_ID`,
`WUSTL_CLIENT_SECRET`, and `WUSTL_API_SCOPE`; it exchanges the institutional
credentials for a short-lived access token at run time. Endpoint URLs must be
absolute HTTPS URLs without embedded credentials, query strings, or fragments.

## Deterministic Scoring

A scenario passes only when all of the following are true:

1. The model returns a final response before the safety cap.
2. Its successful tool calls contain the configured required order as an
   ordered subsequence.
3. Each post-run MCP check satisfies its exact, non-empty, finite, and error
   expectations.
4. Every expected artifact exists.

Tool errors do not automatically fail a scenario because successful recovery
is valid agent behavior. The report records each error code and the subsequent
tool sequence. It does not combine outcomes into an invented scalar quality
score.

The final natural-language response is retained for manual review. It is not
scored through substring matching or another proxy that could pass despite an
incorrect explanation.

## Evidence

The machine-readable report records:

- endpoint, model, requested/effective sampling controls, package versions, and
  tool-catalog SHA-256;
- scenario status, stop reason, final response, ordered tool calls, structured
  error codes, deterministic check failures, and artifact SHA-256 hashes;
- prompt, completion, and total token usage when the endpoint provides it;
- elapsed time as operational metadata, not a quality criterion.

A Markdown rendering summarizes the same record. Raw provider responses and
credentials are not retained.

## Exit Criteria

- Scenario and report schemas have focused tests.
- A fake endpoint exercises success, recovery, cap, and malformed-tool-call
  paths without network access.
- The selected approved endpoint completes the fixed protocol once with the
  selected model and records its effective sampling controls.
- The complete repository test and lint suites pass.
- Results clearly separate deterministic outcome checks from manual review and
  document any endpoint limitations or failed scenarios without rerunning to
  select a better sample.
