# ITU BDS SDSE'24 - Project

This project is part of the Software Development and Software Engineering at ITU. The original project description can be found [here](https://github.com/lasselundstenjensen/itu-sdse-project)

In this project we were tasked with restructuring a Python monolith using the concepts we have learned throughout the course. This project contains a [Dagger](https://github.com/PLtier/github-dagger-workflow-project/blob/main/pipeline.go) and [Github](https://github.com/PLtier/github-dagger-workflow-project/blob/main/.github/workflows/test_action.yml) workflow.

## Project Structure

```
├── README.md          <- Project description
│
├── .dvc
│
├── .github/workflows               <- 
│   ├── tag_version.yml      <- 
│   └── test_action.yml        <- 
│
├── pipeline_deps               <- 
│   └── requirements.txt      <- 
│
├── CODEOWNERS             <-
│
├── go.mod             <-
│
├── go.sum             <-
│
├── pipeline.go             <-
│
├── pyproject.toml             <-
│
├── references         <- 
│
├── requirements.txt   <- 
│
└── github_dagger_workflow_project   <- 
    │
    ├── __init__.py             <- 
    │
    ├── 01_data_transformations.py             <- 
    │
    ├── 02_model_training.py               <- 
    │
    ├── 03_model_selection.py              <- 
    │
    ├── 04_prod_model.py             <- 
    │
    ├── 05_model_deployment.py             <- 
    │
    ├── artifacts
    │   │
    │   └── raw_data.csv.dvc            <- 
    │
    └── utils.py                <- 
```

---


## How to run the code

### Triggering Github Workflow

The workflow can be triggered either by on pull requests to main or manually.

 It can be triggered manually [here](https://github.com/PLtier/github-dagger-workflow-project/actions/workflows/test_action.yml) by pressing `Run workflow` on the `main` branch