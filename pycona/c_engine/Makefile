# Makefile for the pycona C++ acquisition engine.
#
# Builds the three translation units into one shared library that
# c_interface.py loads via ctypes:
#     * Linux / macOS : c_engine.so
#     * Windows (MinGW): c_engine.dll
#
# Usage:
#     make            # optimized build
#     make debug      # unoptimized build with symbols
#     make clean
#
# On Windows without make, use the cross-platform helper instead:
#     python build.py --verify

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall

# Pick the library name + link flags for the host platform.
ifeq ($(OS),Windows_NT)
    TARGET   := c_engine.dll
    LDFLAGS  := -shared -static -static-libgcc -static-libstdc++
else
    UNAME_S := $(shell uname -s)
    TARGET  := c_engine.so
    LDFLAGS := -shared -fPIC
endif

SRCS := c_api.cpp quacq.cpp solver.cpp
HDRS := c_api.h common.h constraint.h constraint_net.h domain.h quacq.h solver.h

.PHONY: all debug clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(SRCS)

debug: CXXFLAGS := -std=c++17 -O0 -g -Wall
debug: clean $(TARGET)

clean:
	-rm -f c_engine.so c_engine.dll *.o
