# math_lab

Private sandbox for mathematical side projects. Doodles live here until they
earn their own repo.

## Projects

| Project | Question | Status | Last touched |
|---|---|---|---|
| [neural-pde-dynamics](projects/neural-pde-dynamics/) | Can learned PDE surrogates capture nonlinear dynamics while respecting known invariants? | active | 2026-11-20 |

Status values: `active`, `parked`, `dead` (with a reason in the journal), `graduated` (link to new repo).

## Layout

```
JOURNAL.md          dated log across all projects, dead ends included
shared/             code used by 2+ projects only; don't put anything here speculatively
projects/<name>/
  README.md         question, current belief, next step
  notes/            derivations (Typst or LaTeX)
  notebooks/        jupytext percent-format .py files (the .ipynb is gitignored)
  src/<pkg>/        importable code
  tests/            math checks: convergence order, invariants, analytic solutions
  experiments/      one dated folder per run: config.yaml + README.md with a verdict
  figures/          only figures you intend to keep
```

## Setup

```bash
uv sync --all-packages
uv run pre-commit install
uv run pytest
```

## Conventions

- Every function that implements an equation names it in the docstring, e.g. `Implements eq. (2) in notes/derivation.typ`.
- Commit configs and seeds. Never commit checkpoints, large arrays, or notebook outputs.
- Every experiment README ends with a one-line verdict: `worked`, `failed because ...`, or `inconclusive`.
- Tag commits that produced a figure you might reuse: `git tag fig/<project>/<short-name>`.

## New experiment

```bash
scripts/new-experiment.sh neural-pde-dynamics kdv-soliton
```

## Graduating a project

When a project is heading toward a paper or patent, needs its own CI, or you'd show it to someone:

```bash
git clone https://github.com/MaximusDecimusV/math_lab neural-pde-dynamics && cd neural-pde-dynamics
git filter-repo --subdirectory-filter projects/neural-pde-dynamics
```

Then mark it `graduated` in the table above with a link.

## IP note

Personal account, personal hardware, private by default. Invention disclosures
are kept separately from this repo; git timestamps are not a reliable legal record.
