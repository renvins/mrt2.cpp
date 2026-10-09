CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra

all: mrt2dump mrt2info

mrt2dump: mrt2dump.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mrt2info: mrt2info.cpp safetensors.cpp json.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

tests/test_mmap: tests/test_mmap.cpp mmap.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^


clean:
	rm -f mrt2dump mrt2info
