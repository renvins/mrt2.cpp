CXX      ?= c++
CXXFLAGS ?= -O2 -Wall -Wextra
# The language standard is kept out of CXXFLAGS, so that
# `make CXXFLAGS=...` (e.g. to add sanitizers) cannot drop it.
STD = -std=c++20

# Shared code, named once. Headers are listed as prerequisites so that
# editing one rebuilds what uses it; $(filter %.cpp,$^) then passes only
# the .cpp files to the compiler.
CORE       = weights.cpp safetensors.cpp json.cpp mmap.cpp profile.cpp \
             weights.h safetensors.h json.h mmap.h profile.h
MODEL_SRC  = ops.cpp ops.h blocks.cpp blocks.h encoder.cpp encoder.h temporal.cpp temporal.h \
             depth.cpp depth.h conv.cpp conv.h codec.cpp codec.h
BUILD      = $(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

all: mrt2 mrt2dump mrt2info

# The program: generate music.
mrt2: mrt2.cpp $(MODEL_SRC) $(CORE)
	$(BUILD)

mrt2dump: mrt2dump.cpp mmap.cpp mmap.h
	$(BUILD)

mrt2info: mrt2info.cpp $(CORE)
	$(BUILD)

tests/test_mmap: tests/test_mmap.cpp mmap.cpp mmap.h
	$(BUILD)

tests/test_weights: tests/test_weights.cpp $(CORE)
	$(BUILD)

tests/test_ops: tests/test_ops.cpp tests/test.h ops.cpp ops.h $(CORE)
	$(BUILD)

tests/test_conv: tests/test_conv.cpp tests/test.h conv.cpp conv.h ops.cpp ops.h profile.cpp profile.h
	$(BUILD)

tests/test_encoder: tests/test_encoder.cpp tests/test.h $(MODEL_SRC) $(CORE)
	$(BUILD)

tests/test_temporal: tests/test_temporal.cpp tests/test.h $(MODEL_SRC) $(CORE)
	$(BUILD)

tests/test_depth: tests/test_depth.cpp tests/test.h $(MODEL_SRC) $(CORE)
	$(BUILD)

tests/test_codec: tests/test_codec.cpp tests/test.h $(MODEL_SRC) $(CORE)
	$(BUILD)

# Speed spike (measurement tool, not the engine): built the way release
# code will be, fully optimized for this machine's CPU.
NATIVE = $(if $(filter arm64 aarch64,$(shell uname -m)),-mcpu=native,-march=native)
tools/spike_bench: tools/spike_bench.cpp ops.cpp ops.h $(CORE)
	$(CXX) $(STD) -O3 $(NATIVE) -Wall -Wextra -o $@ $(filter %.cpp,$^)

spike: tools/spike_bench
	./tools/spike_bench "$(MODEL)"

# Benchmark (step 8): the real model with the profiler on, built the way
# release code will be. `make bench` prints the roofline table.
tools/bench: tools/bench.cpp $(MODEL_SRC) $(CORE)
	$(CXX) $(STD) -O3 $(NATIVE) -DMRT2_PROFILE -Wall -Wextra -o $@ $(filter %.cpp,$^)

bench: tools/bench
	./tools/bench "$(MODEL)"

# The tests read the real model; the ones in REF_TESTS also compare with
# Google's numbers (create those once with `make ref`).
# Override the paths with `make test MODEL=... REF=...`.
MODEL ?= $(HOME)/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors
REF   ?= $(HOME)/models/magenta-rt-v2/ref/ref_small.safetensors
UNIT_TESTS = tests/test_conv
FILE_TESTS = tests/test_mmap tests/test_weights
REF_TESTS  = tests/test_ops tests/test_encoder tests/test_temporal tests/test_depth tests/test_codec
TESTS      = $(UNIT_TESTS) $(FILE_TESTS) $(REF_TESTS)

# Build every test, then run each one; make stops at the first failure.
# Last, check that every weight in the model is a real number.
test: $(TESTS) mrt2info
	@for t in $(UNIT_TESTS); do echo "== $$t"; ./$$t || exit 1; done
	@for t in $(FILE_TESTS); do echo "== $$t"; ./$$t "$(MODEL)" || exit 1; done
	@for t in $(REF_TESTS); do echo "== $$t"; ./$$t "$(MODEL)" "$(REF)" || exit 1; done
	@echo "== mrt2info --verify"; ./mrt2info --verify "$(MODEL)"

# Reference data from Google's JAX model (Python, dev only), then check
# the plan's model spec against it. Re-run only if the dump tool changes.
ref:
	MAGENTA_HOME=$(HOME)/models JAX_PLATFORMS=cpu .venv/bin/python tools/dump_reference.py --out "$(REF)"
	.venv/bin/python tools/check_spec.py "$(REF)" "$(MODEL)"

clean:
	rm -f mrt2 mrt2dump mrt2info $(TESTS) tools/spike_bench tools/bench
	rm -rf *.dSYM tests/*.dSYM tools/*.dSYM

.PHONY: all clean test ref spike bench
