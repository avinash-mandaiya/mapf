#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct GRID 
	{
	int width  = 0;  // x dimension (columns)
	int height = 0;  // y dimension (rows), y=0 at bottom
	int obs    = 0;  // Number of obstacles
	// Column-major occupancy: free[x][y] ∈ {0,1}; 1 = traversable, 0 = blocked.
	std::vector<std::vector<uint8_t>> free;

	inline bool inBounds(int x, int y) const 
		{
		return (0 <= x && x < width && 0 <= y && y < height);
		}

	inline uint8_t at(int x, int y) const { return free[x][y]; }  // free[x][y]
	};

struct AGENT 
	{
	int sx = 0, sy = 0;  // start (math coords: origin bottom-left)
	int gx = 0, gy = 0;  // goal  (math coords)
	};

/*
* Load a MovingAI .map file into bottom-left–origin, column-major grid (free[x][y]).
* Returns true on success; throws std::runtime_error on parse errors.
*/

bool loadMap(const std::string& mapPath, GRID& out);

/*
* Load first K agents from a MovingAI .scen file and convert to math coords.
* (Scenario coordinates are given in file coords (0,0 top-left); we flip y.)
* `mapHeight` is needed for the flip; set it to the corresponding map height.
* If K <= 0, loads all entries in the scen.
* Returns true on success; throws std::runtime_error on parse errors.
*/

bool loadScenario(const std::string& scenPath, int K, int mapHeight, std::vector<AGENT>& agents_out);

/**
* Convenience: load map + first K agents; optionally verify starts/goals are on free cells.
* Throws std::runtime_error on any error (I/O or validation).
*/

bool loadInstance(const std::string& mapPath,
		const std::string& scenPath,
		int K,
		GRID& grid_out,
		std::vector<AGENT>& agents_out,
		bool verify_passable = true);

