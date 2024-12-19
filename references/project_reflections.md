# Reflections

Below is not the part of the documentation.

## A few decisions

- We have noticed a few strong signs an XGBoost was supposed to be in the pipeline. We initially included it, but finally decided on wrapping the code in such a way, that by one-liner one can start effectively compare LR with XGBoost. Please read more in: _Originally posted by @PLtier in [#3 Issues](https://github.com/PLtier/github-dagger-workflow-project/issues/3#issuecomment-2551304436)_.
- We tried to make the code more explicit. E.g. We made `f1_score` to explicitly use `average='binary'` as there were both used `binary` and `weighted` versions.
- LR regression: we realised that code was performing GridSearch but unfortunately it was still saving unfitted LR. We strongly think it was a bug and decided to store best_model output by GridSearch.
- XGBoost: in the initial code `classification_report` for LR regression was computed using `test_data` whereas for Xgboost on `train_data` (!). We changed it to use `test_data`. We strongly think it was an oversight and have changed it.
- XGBoost / LR comparison metric: there was one MLFlow comparison (model selection) using binary F1-Score and also the second one based on weighted F1-Score `model_result` . We decided to use `binary` because it was used in MLFlow comparison which we are certain intent of. (As stated above: we still output always LR).
- We strived to encapsulate as much of the code into functions (no global variables shared except constants). This was to improve
  - readibility
  - better troubleshooting
  - not polluting global namespace (so fewer bugs)
- Imports: we sorted them and removed relative imports i.e. we don't do `import utils` in order not to confuse with an external library.
- The code responsible for registering / transition to staging / deployment has not been deleted (except a few lines) but wrapped and left.
- We moved all paths in scripts to external file.

## What to improve upon

- Take out tests out of the production code as right now we do it.
