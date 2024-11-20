# Classification For Fashion-MNIST dataset

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Comparative performance analysis of, implemented from scratch, Decision Tree, feed-forward NN and other models for Fashion-MNIST dataset.

[](https://cookiecutter-data-science.drivendata.org/opinions/)

## Using the template

### Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         classification_fashion_mnist and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── classification_fashion_mnist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes classification_fashion_mnist a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

### When to refactor notebooks into source code

So we keep notebooks in `notebooks` whereas source code in `classification_fashion_mnist` which is a python module to run.
Don't write code to do the same task in multiple notebooks.

- If it's a data preprocessing task, put it in the pipeline at `data/make_dataset.py` and load data from `data/interim/`.
- If it's useful utility code, refactor it to new module within the doe.

So the structure would look like:

```
...
├── classification_fashion_mnist
│   ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
│   ├── config.py               <- Store useful variables and configuration
│   ├── data
│   │   ├── __init__.py
│   │   └── make_dataset.py      <- Data preprocessing tasks, cleaning, and loading data
│   ├── utils.py                <- General utility functions (e.g., logging, parsing)
│   ├── features.py             <- Feature engineering code
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── predict.py          <- Code for model inference
│   │   └── train.py            <- Code for model training
│   └── plots.py                <- Code to create visualizations
...
```

> **Note**: Classic signs that you are ready to move from a notebook to source code include duplicating old notebooks to start new ones, copy/pasting functions between notebooks, and creating object-oriented classes within notebooks.

It's easy to refactor notebook code because the ccds template makes your project a Python package by default and installs it locally in the requirements file of your chosen environment manager. This enables you to import your project's source code and use it in **notebooks** (like in `notebooks` dir) with a cell like the following:

```python
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code
# in classification_fashion_mnist, it gets loaded
%autoreload 2

from classification_fashion_mnist.data import make_dataset
```

### installing dependencies

First of all, install `make`

Due to simplicity we use `venv`.

```shell
make create_environment # creates .venv
make requirements # fetches & install deps
```

### Naming Convention for Notebooks

We use name notebooks with a scheme that looks like this:

```
0.01-pjb-data-source-1.ipynb
```

- `0.01` - Helps keep work in chronological order. The structure is `PHASE.NOTEBOOK`. `NOTEBOOK` is just the Nth notebook in that phase to be created.
- `pjb` - Your initials; this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
- `data-source-1` - A description of what the notebook covers.

For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:

- `0` - Data exploration - often just for exploratory work.
- `1` - Data cleaning and feature creation - often writes data to `data/processed` or `data/interim`.
- `2` - Visualizations - often writes publication-ready visualizations to `reports`.
- `3` - Modeling - training machine learning models.
- `4` - Publication - Notebooks that get turned directly into reports.

Now that you have your notebook going, start your analysis!

### Branching and commiting

We follow trunk-based branching i.e. for releases we use tags (but in our case - there will be one final release only), we don't work on main but use feature branches.

![Source](./references/trunk1c.png)

We follow
[Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

We prefix the commit message with a type, and then a description.

```bash
feat:     # work related to a feature
docs:     # changes to documentation
fix:      # bug fix
style:    # formatting
refactor: # refactoring and reducing technical debt
test:     # changes related to tests
chore:    # changes that are minor, like updating dependencies
ci:       # changes to pipelines, build systems, etc.
```

A prefix can also be scoped ( feat(scope): description ), and the description can be multiline.

And breaking changes can be highlighted with a ! after the type/scope, e.g. feat!: description.
Though in our project I don't predict these.

```bash
git checkout -b initial-exploration #OK
git push -u origin main #NOT! It's blocked besides
```

### Few practices:
From "Accelerate", Forsgren:
"In short and maybe- counter-intuitively - going faster and releasing more frequently actually LEDs to higher quality products."