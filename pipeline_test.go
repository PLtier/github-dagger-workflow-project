package main

import (
	"testing"
)

func TestRunPipeline(t *testing.T) {
	if err := RunPipeline(true); err != nil {
		t.Fatalf("Pipeline execution failed: %v", err)
	}
}
