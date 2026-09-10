# Building the docs

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`. CI treats warnings as errors and publishes the same build from
`main`.
