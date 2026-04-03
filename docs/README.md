# Documentation

This directory holds project-level documentation that is meant to stay stable
even as scripts and experiments evolve.

## Suggested entry points

- `repo-structure.md`: high-level map of the repository.
- `simulation-workflows.md`: how the main process simulation workflows are run.
- `processes/index.rst`: reference notes for the main process-model modules.
- `demos/index.rst`: comprehensive walkthroughs for demo scripts in `scripts/demos/`.
- `schema/index.rst`: reference notes for schema models and data-prep utilities.
- `utils/index.rst`: utility-module docs for scaling, value updates, plotting, stats, HDF tools, and solver config.
- `experiment/index.rst`: experiment orchestration and training-pipeline docs.
- `surrogates/index.rst`: transport-number surrogate module docs.

## Intended use

Use this folder for documentation that answers one of these questions:

- What is in this repository, and where should I look first?
- How is a typical modeling or validation workflow executed?
- Which scripts produce which outputs?

This folder is a good place for durable technical notes, but not for transient
scratch notes tied to one debugging session.

## Build locally (Sphinx)

From repository root:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Generated site output:

- `docs/_build/html/index.html`

## View locally

After building, open `docs/_build/html/index.html` using one of the options
below.

Open the generated HTML file directly from the terminal:

```bash
# macOS
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html

# Windows PowerShell
start docs/_build/html/index.html
```

If you prefer to serve the docs through a local web server, run:

```bash
python -m http.server 8000 -d docs/_build/html
```

Then open:

- `http://localhost:8000`

The direct file-open commands are the quickest option. The local web server is
the most portable option and can be useful if your browser blocks some local
file behavior.
