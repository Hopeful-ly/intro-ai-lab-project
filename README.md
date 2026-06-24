# CNN Image Colorization

Hello there.

## setup
dependencies management is done with `uv`.
They're already in `pyproject.toml`:
UV can be downloaded from [here](https://docs.astral.sh/uv/getting-started/installation/)
```bash
uv sync
```

## run
To run with different configurations (e.g. other model modes) update them in `config.py` or at the start of `main.py`.
By default, this runs lab classification with the report's best found search parameters.

```bash
uv run python main.py
```

