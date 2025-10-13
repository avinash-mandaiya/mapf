# ---------------- Project ----------------
PROGRAM   := mapf
SRC       := mapf.cpp MapfIO.cpp
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
# Usage:
#   make run map_path=... scen_path=... Zags=... id=... total_time=... beta=... \
#            maxiter=... iterstride=... stoperr=... epsilon=... seed=... numrun=...

# maze-32-32-2.map
# empty-8-8.map

map_path   ?= benchmarks/empty-16-16.map
scen_path  ?= benchmarks/scen-even/empty-16-16-even-1.scen
Zags       ?= 50
id         ?= e45
total_time ?= 30
beta       ?= 0.9
maxiter    ?= 20000
iterstride ?= 100
stoperr    ?= 1e-5
epsilon    ?= 0.01
seed       ?= 10
numrun     ?= 1

run: $(PROGRAM)
	./$(PROGRAM) $(map_path) $(scen_path) $(Zags) $(id) $(total_time) $(beta) $(maxiter) $(iterstride) $(stoperr) $(epsilon) $(seed) $(numrun)

# --------------- Utilities ---------------
clean:
	$(RM) $(PROGRAM) $(OBJ) $(DEPS)

# Debug build with sanitizers
debug: CXXFLAGS := -std=gnu++17 -O0 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -fsanitize=address,undefined -fno-omit-frame-pointer -MMD -MP
debug: LDFLAGS  := -fsanitize=address,undefined
debug: clean all

