# ITU BDS SDSE'24 - Project

This project is part of the Software Development and Software Engineering at ITU. The original project description can be found [here](https://github.com/lasselundstenjensen/itu-sdse-project)

In this project we were tasked with restructuring a Python monolith using the concepts we have learned throughout the course. This project contains a [Dagger](https://github.com/PLtier/github-dagger-workflow-project/blob/main/pipeline.go) and [Github](https://github.com/PLtier/github-dagger-workflow-project/blob/main/.github/workflows/test_action.yml) workflow.

## Project Structure

```
├── README.md                        <- Project description and how to run the code
│
├── .github/workflows                <- Github Action workflows
│   │
│   ├── tag_version.yml              <- Workflow for creating version tags 
│   │
│   └── test_action.yml              <- Workflow that automatically trains and tests model
│
├── pipeline_deps                     
│   │
│   └── requirements.txt             <- Dependencies for the pipeline
│
├── CODEOWNERS                       <- Defines codeowners for the repository
│
├── go.mod                           <- Go file that defines the module and required dependencies
│
├── go.sum                           <- Go file that ensures continuity and integrity of dependencies
│
├── pipeline.go                      <- Dagger workflow written in Go
│
├── pyproject.toml                   <- Configuration file
│
├── .pre-commit-config.yaml          <- Checks quality of code before commits
│
├── Makefile.venv                    <- Creates and manages Pythion virtual enviorment
│
├── references                       <- Documentation and extra resources 
│
├── requirements.txt                 <- Python dependecies need for the project
│
└── github_dagger_workflow_project   <- Source code for the project
    │
    ├── __init__.py                  <- Marks the directory as a Python package
    │
    ├── 01_data_transformations.py   <- Script for data preprocessing and transformation
    │
    ├── 02_model_training.py         <- Script for training the models
    │
    ├── 03_model_selection.py        <- Script for selecting the best perfoming model
    │
    ├── 04_prod_model.py             <- Script for comparing new best model and production model
    │
    ├── 05_model_deployment.py       <- Script for deploying model
    │
    ├── artifacts
    │   │
    │   └── raw_data.csv.dvc         <- Metadata tracked by DVC for data file
    │
    ├── tests
    │   │
    │   └── verify_artifacts.py      <- Tests to check if all artifacts are copied correctly
    │
    └── utils.py                     <- Helper functions
```

---


## How to run the code

### Triggering Github Workflow

The workflow can be triggered either by on pull requests to main or manually.

 It can be triggered manually [here](https://github.com/PLtier/github-dagger-workflow-project/actions/workflows/test_action.yml) by pressing `Run workflow` on the `main` branch