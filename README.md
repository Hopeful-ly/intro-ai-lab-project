# CNN Image Colorization

Hello there.

## setup
dependencies management is done with `uv` (`torch`, `torchvision`, `numpy`,
`matplotlib`). They're already in `pyproject.toml`:

```bash
uv sync
# you can also just use pip, we froze the deps.
pip install -r requirements.txt
```

## run
if you want to change things, update them in `config.py` or at the start of `main.py`
otherwise:
```bash
uv run python main.py
# or
python main.py
```

