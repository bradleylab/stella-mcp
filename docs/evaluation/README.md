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

## Numeric Comparison

When Stella produces a non-empty CSV export, compare it to an MCP/PySD CSV on
the same time grid:

```bash
uv run python -m evaluation.compare_runs \
  results/pysd.csv results/stella.csv \
  --reference-time time \
  --candidate-time Time \
  --column Accumulator=Accumulator \
  --output-json results/evaluation/accumulator-parity.json
```

The comparator performs no interpolation and sets no pass threshold. It rejects
missing, non-numeric, or non-finite values and reports maximum absolute and
relative discrepancies for each explicit column mapping.

## Interpretation

This harness measures deterministic MCP and model workflows. A free-form LLM
agent evaluation would be a separate experiment because it introduces model,
prompt, endpoint, sampling, and scoring choices. Those variables are not part of
the 0.11.0 baseline.
