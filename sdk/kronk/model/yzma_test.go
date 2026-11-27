package model

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/ardanlabs/kronk/sdk/applog"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/hybridgroup/yzma/pkg/llama"
)

func TestYzmaTokenizeNUL(t *testing.T) {
	libPath := libs.Path(os.Getenv("KRONK_TEST_LIB_PATH"))
	modelPath := installedTestModel(t, "KRONK_TEST_MODEL", "Qwen3-0.6B-Q8_0.gguf")

	if err := llama.Load(libPath); err != nil {
		t.Fatalf("load llama.cpp: %v", err)
	}
	if err := llama.Init(); err != nil {
		t.Fatalf("initialize llama.cpp: %v", err)
	}

	params := llama.ModelDefaultParams()
	params.VocabOnly = 1
	mdl, err := llama.ModelLoadFromFile(modelPath, params)
	if err != nil {
		t.Fatalf("load model vocabulary: %v", err)
	}
	t.Cleanup(func() {
		if err := llama.ModelFree(mdl); err != nil {
			t.Errorf("free model: %v", err)
		}
	})

	vocab := llama.ModelGetVocab(mdl)
	tokens := llama.Tokenize(vocab, "left\x00right", false, true)
	if len(tokens) == 0 {
		t.Fatal("tokenize embedded NUL: got no tokens")
	}
	if prefix := llama.Tokenize(vocab, "left", false, true); slices.Equal(tokens, prefix) {
		t.Fatalf("tokenize embedded NUL: got prefix tokens %v; bytes after NUL had no effect", tokens)
	}
}

func TestLoadModelFromFilesCancelledBeforeLoad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := loadModelFromFiles(ctx, applog.DiscardLogger, []string{"model.gguf"}, llama.ModelParams{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("load with a cancelled context: got %v, want %v", err, context.Canceled)
	}
}

func TestLoadModelFromFilesCancelledDuringLoad(t *testing.T) {
	libPath := libs.Path(os.Getenv("KRONK_TEST_LIB_PATH"))
	modelPath := installedTestModel(t, "KRONK_TEST_MODEL", "Qwen3-0.6B-Q8_0.gguf")

	if err := llama.Load(libPath); err != nil {
		t.Fatalf("load llama.cpp: %v", err)
	}
	if err := llama.Init(); err != nil {
		t.Fatalf("initialize llama.cpp: %v", err)
	}

	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &cancelledAfterFirstCheck{Context: base}

	mdl, err := loadModelFromFiles(ctx, applog.DiscardLogger, []string{modelPath}, llama.ModelDefaultParams())
	if err == nil {
		llama.ModelFree(mdl)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("load cancelled after it started: got %v, want %v", err, context.Canceled)
	}
}

// cancelledAfterFirstCheck is live for its first Err call and cancelled from
// then on, so a load starts and must stop at llama.cpp's next progress report.
type cancelledAfterFirstCheck struct {
	context.Context
	checks atomic.Int32
}

func (ctx *cancelledAfterFirstCheck) Err() error {
	if ctx.checks.Add(1) > 1 {
		return context.Canceled
	}
	return nil
}

// installedTestModel resolves an installed fixture unless an explicit path is
// supplied. Missing fixtures fail in CI and remain optional for local runs.
func installedTestModel(t *testing.T, env, name string) string {
	t.Helper()
	if file := os.Getenv(env); file != "" {
		return file
	}
	pattern := filepath.Join(defaults.BaseDir(""), "models", "*", "*", name)
	files, err := filepath.Glob(pattern)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		if os.Getenv("CI") == "true" {
			t.Fatalf("required model %s not installed", name)
		}
		t.Skipf("model %s not installed", name)
	}
	return files[0]
}
