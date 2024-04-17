# Bayesian evaluation of climate simulations

![main](https://github.com/markusritschel/bayes-climsim-eval/actions/workflows/main.yml/badge.svg)

![License MIT license](https://img.shields.io/github/license/markusritschel/bayes-climsim-eval)



Evaluation of climate model simulations (e.g. from CMIP) with a Bayesian approach

## <u>Table of Contents <!-- omit in toc --></u>
- [Bayesian evaluation of climate simulations](#bayesian-evaluation-of-climate-simulations)
  - [Table of Contents ](#table-of-contents-)
  - [Preparation](#preparation)
    - [Cloning the project to your local machine](#cloning-the-project-to-your-local-machine)
    - [Setting up a dedicated virtual environment](#setting-up-a-dedicated-virtual-environment)
    - [Installing requirements](#installing-requirements)
    - [Make raw data available](#make-raw-data-available)
  - [High-level \& Low-level Code](#high-level--low-level-code)
  - [Testing](#testing)
  - [Project Structure](#project-structure)
  - [Dummy files](#dummy-files)
  - [Maintainer](#maintainer)
  - [Contact \& Issues](#contact--issues)

## Preparation
### Cloning the project to your local machine
To reproduce the project, clone this repository on your machine
```bash
git clone https://github.com/markusritschel/bayes-climsim-eval
```

### Setting up a dedicated virtual environment
As a next step, although optional, I'd highly recommend that you create a new virtual environment. <br>
> **Note**:
> I recommend that you use [Conda](https://docs.conda.io/en/latest/miniconda.html) as a package manager. For performance boost, it is recommended to use [Mamba](https://mamba.readthedocs.io/).

You can simply use the Makefile command `make setup-conda-env` from inside the cloned directory (`cd bayes-climsim-eval/`). 
This is probably the easiest way to get set up, especially if you're not familiar with virtual environments.
This will install Mamba, create a new conda environment with the same name as the project directory, install the packages as they are listed in the environment.yml, and activate the environment.

### Installing requirements
Then, in the directory you just cloned run either `python setup.py install` or simply `make src-available` to make the source code in `src` available as a package. 
From now on you can use `import src` in any python context within your conda environment.

> 👉 **Note:** *If* you intend to make changes to the code and want them reflected in the installed instance, replace the `install` in the previous command with `develop`:
> ```bash
> python setup.py develop
> ```
> The command `make src-available` will actually use the `develop` option.
> [See [here](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for an explanation]

If you don't wanna use conda for any reason, you can also install the required packages via pip only:
```bash
pip install -r requirements.txt
```

> **Note**:
> If you experience that something is not working (e.g. creating the documentation via `make docs`) try to perform an update via `mamba update --all`. This might solve the problem.

### Make raw data available
Next, make the raw data available or accessible under `data/` (see project structure below).
If the project is dealing with large amounts of data that reside somewhere outside your home directory,
I would suggest that you link the respective subdirectories inside `data/` accordingly.
The python scripts should be able to follow symlinks.

<!-- If all is set up, you can run `make test-structure` to perform some tests before starting running the scripts or Jupyter notebooks in the respective directories. -->


## High-level & Low-level Code
All _high-level_ code (i.e. the code that the user is directly interacting with) resides in the `scripts/` and the `notebooks/` directory.
High-level code is, for example, code that produces a figure, a report, or similar.\
Both the scripts and the notebooks should be named in a self-explanatory way that indicates their order of execution and their purpose.

Code residing in `src/` is _exclusively_ source code or _low-level_ code and is not actively executed.
<!-- For standard tasks, you might find respective commands in the Makefile. Just type `make` to see a list of available commands. -->

<u>A recommendation for long-running tasks:</u><br>
Some tasks like data processing will need a long time. 
It is highly recommended that you use a detachable terminal environment like `screen` or [`tmux`](https://github.com/tmux/tmux/wiki).
This way you can detach from the session (even close your terminal) without losing or ending the process.
Alternatively, if you work on a high-performance computer, make use of the queuing system to submit jobs.


## Testing
To test your code, run `make tests` in the root directory.
This will execute both the unit tests and docstring examples using `pytest`.

<!-- Run `make coverage` to generate a test coverage report and `make lint` to check code style consistency. -->


## Project Structure

    ├── assets             <- A place for assets like shapefiles or config files
    │
    ├── data               <- Contains all data used for the analyses in this project.
    │   │                     The sub-directories can be links to the actual location of your data.
    │   │                     However, they should never be under version control! (-> .gitignore)
    │   ├── interim        <- Intermediate data that have been transformed from the raw data
    │   ├── processed      <- The final, processed data used for the actual analyses
    │   └── raw            <- The original, immutable(!) data
    │
    ├── docsrc             <- The technical documentation (default engine: Sphinx; but feel free to use 
    │                         MkDocs, Jupyter-Book or anything similar).
    │                         This should contain only documentation of the code and the assets.
    │                         A report of the actual project should be placed in `reports/book`.
    │
    ├── logs               <- Storage location for the log files being generated by scripts (tip: use `lnav` to view)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     and a short `-` or `_` delimited description, e.g. `01-initial-analyses`
    │   ├── _paired        <- Optional location for your paired jupyter notebook files
    │   ├── exploratory    <- Notebooks for exploratory tasks
    │   └── reports        <- Notebooks generating reports and figures
    │
    ├── references         <- Data descriptions, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated reports (e.g. HTML, PDF, LaTeX, etc.)
    │   ├── book           <- A Jupyter-Book describing the project
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── scripts            <- High-level scripts that use (low-level) source code from `src/`
    ├── src                <- Source code (and only source code!) for use in this project
    │   ├── tests          <- Contains tests for the code in `src/`
    │   └── __init__.py    <- Makes src a Python module and provides some standard variables
    │
    ├── .env               <- In this file, specify all your custom environment variables
    │                         Keep this out of version control!
    ├── CHANGELOG.md       <- All major changes should go in there
    ├── jupytext.toml      <- Configuration file for jupytext
    ├── LICENSE            <- The license used for this project
    ├── Makefile           <- A self-documenting Makefile for standard CLI tasks
    ├── README.md          <- The top-level README of this project
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── setup.py           <- Setup python file to install your source code in your (virtual) python environment



## Dummy files
The following files are for demonstration purposes only and can be safely deleted if not needed:

    ├── notebooks/01-minimal-example.ipynb
    ├── docsrc/source/*
    ├── reports/book/*
    ├── scripts/01-test.py
    └── src
        ├── tests/*
        └── submodule.py



## Maintainer
- [markusritschel](https://github.com/markusritschel)


## Contact & Issues
For any questions or issues, please contact me via git@markusritschel.de or open an [issue](https://github.com/markusritschel/bayes-climsim-eval/issues).


---
&copy; [Markus Ritschel](https://github.com/markusritschel), 2024
