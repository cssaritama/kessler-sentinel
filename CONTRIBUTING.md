# Contributing

## How this repository is run

Profile `product`. Data class Public. State Active. Owner Carlos Saritama.
README.md states what is deployed and what is not.

Before starting work, read:

1. `README.md`
2. `predict.py`
3. `tests/test_model.py`
4. `docs/standards/PROJECT_STANDARDS.md` and `docs/standards/PROFILE.md`

Then follow links only as far as the change needs. Do not read `archive/`, `past_version/`, `.venv/` and `.pytest_cache/`
unless the change needs that evidence.

Rules:

- Orbital quantities keep their units (km, degrees, seconds) through every feature and prediction.
- The training target is generated from the synthetic inputs by `src/build_cdm_dataset.py`. A model meant for real use needs observed outcomes, not a formula of its inputs.
- Tests run from `tests/` only; archived folders are excluded from collection.
- Never commit API keys, credentials or private tokens.
- `docs/standards/` is a verified copy of the Systems Studio Engineering Standard. Do not edit it; start specifications, decisions and verification records from `docs/standards/templates/`.
- Work on a focused branch and open a pull request. Keep `main` working.

Verify:

```bash
make test
```
