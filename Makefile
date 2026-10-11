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

mrt2info: mrt2info.cpp mmap.cpp safetensors.cpp json.cpp mmap.h safetensors.h json.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

tests/test_mmap: tests/test_mmap.cpp mmap.cpp mmap.h
	$(CXX) $(STD) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)

clean:
	rm -f mrt2dump mrt2info tests/test_mmap tests/test_weights
	rm -rf *.dSYM tests/*.dSYM

.PHONY: all clean
