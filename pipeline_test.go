package main

import (
	"testing"
)

// It checkes whether all of the necessary artifacts are generated (the last step). It's supposed to be used locally, not in CI/CD.
func TestPipelineAndAllArtifacts(t *testing.T) {
	if err := RunPipeline(true); err != nil {
		t.Fatalf("Pipeline execution failed: %v", err)
	}
}
