# ---------------- Project ----------------
PROGRAM   := mapf
SRC       := mapf.cpp
OBJ       := $(SRC:.cpp=.o)
DEPS      := $(OBJ:.o=.d)

# ---------------- Toolchain --------------
CXX       ?= g++
CPPFLAGS  := -I.                               # where ArrayND.hpp lives
WFLAGS	  := #-Wall -Wextra -Wpedantic -Wshadow -Wconversion 
CXXFLAGS  := -std=gnu++17 -O2 -MMD -MP
LDFLAGS   :=
LDLIBS    :=

# ---------------- Targets ----------------
.PHONY: all run clean debug

all: $(PROGRAM)

$(PROGRAM): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Rebuild when ArrayND.hpp changes (and any other headers via -MMD)
%.o: %.cpp ArrayND.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WFLAGS) -c $< -o $@

-include $(DEPS)

# --------------- Run args ----------------
# Usage: make run input_file=... id=... beta=... maxiter=... iterstride=... stoperr=... epsilon=... seed=... numrun=...
input_file ?= instance3.txt
id         ?= t3
total_time = 8
beta       ?= 0.9
maxiter    ?= 50000
iterstride ?= 100
stoperr    ?= 1e-5
epsilon    ?= 0.000
seed       ?= 2
numrun     ?= 1

run: $(PROGRAM)
	./$(PROGRAM) $(input_file) $(id) $(total_time) $(beta) $(maxiter) $(iterstride) $(stoperr) $(epsilon) $(seed) $(numrun)

# --------------- Utilities ---------------
clean:
	$(RM) $(PROGRAM) $(OBJ) $(DEPS)

# Debug build with sanitizers
debug: CXXFLAGS := -std=gnu++17 -O0 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -fsanitize=address,undefined -fno-omit-frame-pointer -MMD -MP
debug: LDFLAGS  := -fsanitize=address,undefined
debug: clean all

