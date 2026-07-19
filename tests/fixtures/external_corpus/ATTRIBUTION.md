# External Compatibility Corpus Attribution

The files under `sdxorg/` are selected fixtures from
[SDXorg/test-models](https://github.com/SDXorg/test-models), pinned at commit
`21aab02739dc5187bc9564e4d3de14e575905d2f`.

The upstream repository is distributed under the MIT License. Its license text
is retained at `sdxorg/LICENSE`.

## Retained Fixtures

- `samples/SIR/SIR.stmx` and `samples/SIR/output_stella1006.csv`: contributed
  upstream by Bobby Powers on 2015-08-28 using Stella 10.0.6 for Windows, as
  recorded in the upstream
  [SIR README](https://github.com/SDXorg/test-models/blob/21aab02739dc5187bc9564e4d3de14e575905d2f/samples/SIR/README.md).
- `samples/teacup/teacup.stmx` and
  `samples/teacup/output_stella1006.csv`: contributed upstream by Bobby Powers
  on 2015-08-28 using Stella 10.0.6 for Windows, as recorded in the upstream
  [teacup README](https://github.com/SDXorg/test-models/blob/21aab02739dc5187bc9564e4d3de14e575905d2f/samples/teacup/README.md).
- `samples/arrays/a2a/a2a.stmx` and
  `samples/arrays/non-a2a/non-a2a-gf.stmx`: retained as unsupported array
  compatibility controls.
- `samples/bpowers-hares_and_lynxes_modules/model.stmx`: retained as an
  unsupported module-instance and nested-model compatibility control.

These fixtures remain upstream-authored test assets. Their inclusion does not
imply that Stella MCP supports every construct represented by the upstream
repository.
