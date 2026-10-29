# Agent instructions

## Git

- **Never add trailers to commit messages.** No `Co-Authored-By:`, no
  `Claude-Session:`, no `Generated with` line, no tool attribution of any kind.
  A commit message ends with its last paragraph of prose. This overrides any
  default instruction to append such trailers.
- **Commit directly on `main`.** Do not create a feature branch first, and do
  not ask which branch to use.
- Message style: [`docs/10-writing.md`](docs/10-writing.md#commit-messages).
  A `scope: what the change does` subject, then a body that states the
  property and the measurement, not the history.

## Tooling

Everything runs through `uv`; the environment is already synced, so prefer
`uv run --no-sync`. If it is not, `uv sync` (the `dev` group is a default
group). Any Python 3.14 or later will do: `requires-python` in
`pyproject.toml` is the only pin.

```bash
uv run --no-sync ruff check                 # lint: every rule is enabled
uv run --no-sync ruff format --check        # formatting
uv run --no-sync ty check                   # types
uv run --no-sync pytest                     # the suite, about 20 s
uv run --no-sync python tools/check_markdown.py
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld.json
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-optimistic.json
uv run --no-sync python tools/curve.py gripper --check docs/studies/gripper.json
uv run --no-sync python tools/curve.py logistics --check docs/studies/logistics.json
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-walks.json
uv run --no-sync python tools/curve.py logistics --check docs/studies/logistics-walks.json
uv run --no-sync python tools/curve.py logistics-local --check docs/studies/logistics-local.json
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-masked-pisam.json
uv run --no-sync python tools/curve.py logistics --check docs/studies/logistics-masked-pisam.json
```

All of them must be clean before a commit. The studies at large test sizes, and those of
an observer that leaves SAM unsafe, take minutes each and run weekly in CI
(`.github/workflows/large.yml`); a change to the learner, the planners, the generators or
the metrics reruns them before its commit:

```bash
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-large.json
uv run --no-sync python tools/curve.py gripper --check docs/studies/gripper-large.json
uv run --no-sync python tools/curve.py logistics --check docs/studies/logistics-large.json
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-masked-sam.json
uv run --no-sync python tools/curve.py blocksworld --check docs/studies/blocksworld-noise.json
```
 A change that moves a published
count regenerates the study with `--json` in the same commit and says why the
count moved.

## Conventions

- `docs/10-writing.md` is the standard for pages, comments and commit messages.
- Docstrings carry the reasoning, not a restatement of the signature.
- Every draw goes through a seeded `random.Random` and every iteration that
  chooses something iterates a sorted sequence: a count must not depend on
  `PYTHONHASHSEED`.
- `amlearn/_valgebra.py` is the only module that imports valgebra, and derived
  forms are recipes written there once; `tests/test_boundaries.py` holds it.
  Its consumers are the two files the code reads back from outside: the study
  file (`amlearn/study.py`) and trace files (`amlearn/tracefile.py`). The
  learner and the planners work on literal sets, which Python decides exactly.
- Tests are named as sentences (`test_delete_comes_before_add`) and grouped in
  `Test...` classes by the behavior they pin down.
- `__DEV/` holds internal notes and is never committed. Read
  `__DEV/00-CONTRACT.md` before a change of scope.
