# Building the docs

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Open `_build/html/index.html`. CI builds on every pull request and deploys to GitHub Pages on push to `main`. Warnings are errors.
