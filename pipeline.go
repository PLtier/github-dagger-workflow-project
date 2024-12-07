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

	container := client.Container().From("python:3.12.2-bookworm")

	// print
	ls(ctx, container)
	// _, err = python.File("../artifacts/lead_model_lr.pkl").Export(ctx, "model.pkl")
	container = copyCode(client, container)
	container = installDeps(container)
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

func ls(ctx context.Context, container *dagger.Container, message string) {
	info, err := container.WithExec([]string{"ls", "-la"}).Stdout(ctx)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
	fmt.Println(info)
}
