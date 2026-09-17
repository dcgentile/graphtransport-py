# graphtransport

Python port of [GraphTransportation.jl](../GraphTransportation.jl), aimed at being
usable directly in machine learning pipelines (e.g. as a differentiable layer or
graph/dataset utility), not just a 1:1 translation of the Julia API.

## Status

Steps 1-2 done: package skeleton, Markov chain constructors, `MarkovGraph`.
See [PORTING_PLAN.md](PORTING_PLAN.md) for the up-to-date, in-depth plan for
everything remaining (re-derived from the current Julia source, which has
grown substantially since Step 2). Each step lands as its own reviewed,
tested commit/PR — nothing is ported in bulk.

There is a `graphtransport-py-draft` sibling directory containing an earlier,
unreviewed first attempt at a full port. It is kept only as reference material
and is not part of this package's history or design.

## Development

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
