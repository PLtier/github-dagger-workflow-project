package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	if err := RunPipeline(false); err != nil {
		fmt.Printf("Pipeline execution failed: %v", err)
	}
}
func RunPipeline(isTesting bool) error {
	ctx := context.Background()
	client, err := dagger.Connect(ctx)
	if err != nil {
		return fmt.Errorf("failed to connect to Dagger: %v", err)
	}
	defer client.Close()
	pipCache := client.CacheVolume("pip_cache")
	container := client.Container().From("python:3.11.9-bookworm").
		WithMountedCache("/root/.cache/pip", pipCache)

	container = container.
		WithExec([]string{"mkdir", "pipeline"}).
		WithWorkdir("/pipeline")
	container = copyCode(client, container)
	container = installDeps(container)

	container = container.
		WithWorkdir("/pipeline/github_dagger_workflow_project")
	container = pullData(container)
	container = executeTransformations(container)
	container = executeTraining(container)
	container = executeSelection(container)
	if isTesting {
		_, err = container.
			WithFile("./tests/verify_artifacts.py", client.Host().File("tests/verify_artifacts.py")).
			WithExec([]string{"python", "./tests/verify_artifacts.py"}).
			Stderr(ctx)
	} else {
		err = retrieveArtifacts(ctx, container)
	}
	container = container.WithWorkdir("/")
	if err != nil {
		return fmt.Errorf("failed to run pipeline: %v", err)
	}
	return nil

}

func copyCode(client *dagger.Client, container *dagger.Container) *dagger.Container {
	host := client.Host()
	container = container.
		WithDirectory("./.dvc", host.Directory(".dvc")).
		WithDirectory("./.git", host.Directory(".git")).
		WithFile("./pyproject.toml", host.File("pyproject.toml")).
		WithFile("./requirements.txt", host.File("pipeline_deps/requirements.txt")).
		WithDirectory("./github_dagger_workflow_project", host.Directory("github_dagger_workflow_project"))
	return container
}

func installDeps(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"pip", "install", "-r", "./requirements.txt"})
	return container
}

func pullData(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"dvc", "update", "./artifacts/raw_data.csv.dvc"})
	return container
}

func executeTransformations(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"python", "./01_data_transformations.py"})
	return container
}

func executeTraining(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"python", "./02_model_training.py"})
	return container
}

func executeSelection(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"python", "./03_model_selection.py"})
	return container
}

func retrieveArtifacts(ctx context.Context, container *dagger.Container) error {
	filesToExport := map[string]string{
		"./artifacts/best_experiment.pkl":     "artifacts/best_experiment.pkl",
		"./artifacts/cat_missing_impute.csv":  "artifacts/cat_missing_impute.csv",
		"./artifacts/columns_drift.json":      "artifacts/columns_drift.json",
		"./artifacts/columns_list.json":       "artifacts/columns_list.json",
		"./artifacts/date_limits.json":        "artifacts/date_limits.json",
		"./artifacts/lead_model_lr.pkl":       "artifacts/lr_model.pkl",
		"./artifacts/best_model.pkl":          "artifacts/model.pkl",
		"./artifacts/model_results.json":      "artifacts/model_results.json",
		"./artifacts/outlier_summary.csv":     "artifacts/outlier_summary.csv",
		"./artifacts/raw_data.csv":            "artifacts/raw_data.csv",
		"./artifacts/scaler.pkl":              "artifacts/scaler.pkl",
		"./artifacts/training_data.csv":       "artifacts/train_data.csv",
		"./artifacts/train_data_gold.csv":     "artifacts/train_data_gold.csv",
		"./artifacts/lead_model_xgboost.json": "artifacts/xgboost_model.json",
		"./artifacts/lead_model_xgboost.pkl":  "artifacts/xgboost_model.pkl",
	}

	for srcPath, outputName := range filesToExport {
		_, err := container.
			File(srcPath).
			Export(ctx, outputName)
		if err != nil {
			return fmt.Errorf("failed to retrieve and rename file %s: %v", srcPath, err)
		}
	}
	return nil
}

func ls(ctx context.Context, container *dagger.Container, dir string, message string) {
	fmt.Println(message)
	info, err := container.WithWorkdir(dir).WithExec([]string{"ls", "-la"}).Stdout(ctx)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
	fmt.Println(info)

}
