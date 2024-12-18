package main

import (
	"testing"
)

// It checkes whether all of the necessary artifacts are generated (the last step)
func TestPipelineAndAllArtifacts(t *testing.T) {
	if err := RunPipeline(true); err != nil {
		t.Fatalf("Pipeline execution failed: %v", err)
	}
}
