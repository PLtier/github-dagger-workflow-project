package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()
	client, err := dagger.Connect(ctx)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
	defer client.Close()
	pipCache := client.CacheVolume("pip_cache")
	container := client.Container().From("python:3.12.2-bookworm").WithMountedCache("/root/.cache/pip", pipCache)

	// _, err = python.File("../artifacts/lead_model_lr.pkl").Export(ctx, "model.pkl")
	container = copyCode(client, container)
	container = installDeps(container)
	container = pullData(container)
	ls(ctx, container, "/pipeline/github_dagger_workflow_project/artifacts", "After copying code and installing dependencies")
}

func copyCode(client *dagger.Client, container *dagger.Container) *dagger.Container {
	container = container.
		WithDirectory("/pipeline/.dvc", client.Host().Directory(".dvc")).
		WithDirectory("/pipeline/.git", client.Host().Directory(".git")).
		WithFile("/pipeline/requirements.txt", client.Host().File("pipeline_deps/requirements.txt")).
		WithDirectory("/pipeline/github_dagger_workflow_project", client.Host().Directory("github_dagger_workflow_project"))
	return container
}

func installDeps(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"pip", "install", "-r", "pipeline/requirements.txt"})
	return container
}

func pullData(container *dagger.Container) *dagger.Container {
	container = container.WithWorkdir("/pipeline")
	container = container.WithExec([]string{"dvc", "update", "github_dagger_workflow_project/artifacts/raw_data.csv.dvc"})
	container = container.WithWorkdir("/")
	return container
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
