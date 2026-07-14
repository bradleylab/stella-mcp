# Evaluation Guide

## Deterministic Baseline

From a source checkout with `uv` installed:

```bash
uv run --extra sim python -m evaluation.runner \
  --require sim \
  --artifact-dir /private/tmp/stella-mcp-evaluation/artifacts \
  --output-json results/evaluation/0.11.0-baseline.json \
  --output-markdown results/evaluation/0.11.0-baseline.md
```

Use `--scenario ID` repeatedly to run a subset. Omitting the `sim` extra skips
only scenarios that declare that capability; `--require sim` converts that
condition into a startup failure.

Reports include the tool-catalog hash and generated artifact hashes. Tool result
text replaces repository and artifact-directory paths with stable tokens so the
record does not retain local machine paths.

## Desktop Workflow

Candidate `.stmx` artifacts are opened, run, and saved in Stella Professional.
Accepted files are copied to `tests/fixtures/compat_corpus/` and registered in
its `manifest.json` with application provenance and visual notes. Check manifest
integrity with:

```bash
uv run python scripts/sync_compat_corpus_manifest.py --check
```

The screenshot below records the connector-layout issue observed in the SIR
fixture under Stella Professional 4.1.1.

![SIR model open in Stella Professional 4.1.1](images/stella-4.1.1-sir.jpg)

The corrected built-in template and its Stella-saved fixture are documented in
the [SIR layout follow-up](2026-07-13-sir-layout-followup.md).

## Numeric Comparison

The first desktop comparison is documented in
[`2026-07-13-numeric-parity.md`](2026-07-13-numeric-parity.md). Reproduce its
PySD run and machine-readable report with:

```bash
uv run --extra sim python -m evaluation.desktop_parity \
  tests/fixtures/compat_corpus/stella_4_1_1_accumulator.stmx \
  results/evaluation/0.12.0-accumulator-stella.csv \
  --pysd-output results/evaluation/0.12.0-accumulator-pysd.csv \
  --output-json results/evaluation/0.12.0-accumulator-parity.json \
  --stella-version 4.1.1 \
  --stella-time Years \
  --column Accumulator=Accumulator \
  --column input=input \
  --column rate=rate
```

The orchestration command records artifact hashes and engine versions. Its
underlying comparator performs no interpolation and sets no pass threshold. It
rejects missing, non-numeric, or non-finite values and reports maximum absolute
and relative discrepancies for each explicit column mapping.

## Interpretation

This harness measures deterministic MCP and model workflows. A free-form LLM
agent evaluation would be a separate experiment because it introduces model,
prompt, endpoint, sampling, and scoring choices. Those variables are not part of
the 0.11.0 baseline.

## Free-Form Agent Protocol

The separate protocol is specified in
[`../plans/2026-07-13-free-form-agent-evaluation-spec.md`](../plans/2026-07-13-free-form-agent-evaluation-spec.md)
and versioned in `evaluation/agent_scenarios.json`. From a source checkout, run
it with an explicitly approved provider, model, and supported sampling mode:

```bash
uv run --group agent-eval --extra sim \
  python -m evaluation.run_agent_evaluation \
  --provider PROVIDER \
  --model MODEL_ID \
  --sampling-mode SAMPLING_MODE \
  --artifact-dir results/evaluation/0.12.0-agent-artifacts \
  --output-json results/evaluation/0.12.0-agent-evaluation.json \
  --output-markdown results/evaluation/0.12.0-agent-evaluation.md
```

`PROVIDER` is `openai` or `washu`. `SAMPLING_MODE` declares which of the
protocol's requested `temperature` and `seed` fields the chosen endpoint/model
supports: `both`, `temperature`, `seed`, or `none`. The command refuses to
replace existing expected artifacts or result files.
