package kronk

import (
	"sync"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/hybridgroup/yzma/pkg/utils"
)

func TestInitLogCallback(t *testing.T) {
	var mu sync.Mutex
	var logged []string
	cb := purego.NewCallback(func(level int32, text *byte, _ uintptr) uintptr {
		mu.Lock()
		defer mu.Unlock()
		logged = append(logged, utils.BytePtrToString(text))
		return 0
	})

	if err := Init(WithLogCallback(cb)); err != nil {
		t.Skipf("llama.cpp libraries not installed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(logged) == 0 {
		t.Fatal("nothing llama.cpp logged during Init reached the callback")
	}
}
