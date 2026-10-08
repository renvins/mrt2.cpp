CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra

mrt2dump: mrt2dump.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<