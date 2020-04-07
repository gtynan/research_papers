# Research papers

Throughout researching disertation found useful research papers that either:

1. Did not have a repository linked.
2. Link to repository was dead. 
3. Not written in Python. 

Aim of this project is to reproduce said papers before considering them as sources for disertation.

## Installation

### Virtual environment

The virtaul environemnt used was [poetry](https://python-poetry.org/docs/) 

Can be `pip` installed using:

```shell
pip install --user poetry
```

#### Creating environment

```shell
poetry install
```

This should read the `pyproject.toml`, use the versions stated in `poetry.lock` and install

#### Activating environment

```shell
poetry shell
```

## Usage

Every paper pipeline was developed using [kedro](https://kedro.readthedocs.io/en/stable/)

### Running a pipeline

```shell
cd PAPER_INTERESTED_IN
kedro run
```

## Papers covered

1. Dynamic Bradley–Terry modelling of sports tournaments.
    - **Paper link**: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9876.2012.01046.x 
    - **Folder**: `dynamic_bt/`


## Pipeline structure

### conf

### data

### src