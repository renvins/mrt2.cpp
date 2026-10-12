CXX      ?= c++
CXXFLAGS ?= -O2 -Wall -Wextra
# The language standard is kept out of CXXFLAGS, so that
# `make CXXFLAGS=...` (e.g. to add sanitizers) cannot drop it.
STD = -std=c++20

all: mrt2dump mrt2info

# Headers are listed as prerequisites so that editing one rebuilds the
# program. $(filter %.cpp,$^) passes only the .cpp files to the compiler.
mrt2dump: mrt2dump.cpp mmap.cpp mmap.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

mrt2info: mrt2info.cpp weights.cpp mmap.cpp safetensors.cpp json.cpp \
          weights.h mmap.h safetensors.h json.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

tests/test_mmap: tests/test_mmap.cpp mmap.cpp mmap.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

tests/test_weights: tests/test_weights.cpp weights.cpp safetensors.cpp json.cpp mmap.cpp \
                    weights.h safetensors.h json.h mmap.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

# The tests read the real model. Override with `make test MODEL=...`.
MODEL ?= $(HOME)/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors
TESTS = tests/test_mmap tests/test_weights

# Build every test, then run each one; make stops at the first failure.
# Last, check that every weight in the model is a real number.
test: $(TESTS) mrt2info
	@for t in $(TESTS); do echo "== $$t"; ./$$t "$(MODEL)" || exit 1; done
	@echo "== mrt2info --verify"; ./mrt2info --verify "$(MODEL)"

clean:
	rm -f mrt2dump mrt2info $(TESTS)
	rm -rf *.dSYM tests/*.dSYM

.PHONY: all clean test
