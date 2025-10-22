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
map_id     = maze-32-32-2 
scenario   = 1

map_path   ?= benchmarks/maze-32-32-2.map
scen_path  ?= benchmarks/scen-even/maze-32-32-2-even-1.scen
Zags       ?= 3
id         ?= x28
Ztime      ?= 80
Total_Cost ?= 119
beta       ?= 0.9
maxiter    ?= 100000
iterstride ?= 200
stoperr    ?= 1e-8
epsilon    ?= 0.05
seed       ?= 10
numrun     ?= 1

run: $(PROGRAM)
	./$(PROGRAM) $(map_path) $(scen_path) $(Zags) $(id) $(Ztime) $(Total_Cost) $(beta) $(maxiter) $(iterstride) $(stoperr) $(epsilon) $(seed) $(numrun)

# --------------- Utilities ---------------
clean:
	$(RM) $(PROGRAM) $(OBJ) $(DEPS)

# Debug build with sanitizers
debug: CXXFLAGS := -std=gnu++17 -O0 -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -fsanitize=address,undefined -fno-omit-frame-pointer -MMD -MP
debug: LDFLAGS  := -fsanitize=address,undefined
debug: clean all

