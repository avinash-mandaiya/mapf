#include "MapfIO.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

static inline void rstrip_cr(std::string& s) 
	{
	if (!s.empty() && s.back() == '\r') s.pop_back(); // tolerate CRLF files
	}

static inline int parse_kv_int(const std::string& line, const char* expect_key) 
	{
	std::istringstream iss(line);
	std::string key; int value = 0;
	if (!(iss >> key >> value) || key != expect_key) 
		throw std::runtime_error(std::string("Bad header line, expected '") + expect_key + " N': " + line);
	return value;
	}

// MovingAI: treat '.' and 'G' as traversable for grid planning.
static inline bool is_free_char(char c) { return (c == '.' || c == 'G'); }

// Convert file coords (0,0 top-left; y down) -> math coords (0,0 bottom-left; y up).
static inline std::pair<int,int> file_to_math(int x_file, int y_file, int H) 
	{
	return { x_file, H - 1 - y_file };
	}

bool loadMap(const std::string& mapPath, GRID& out) 
	{
	std::ifstream in(mapPath);
	if (!in) return false;

	std::string line;

	// "type octile" (sanity check "type")
	if (!std::getline(in, line)) return false; rstrip_cr(line);
	if (line.rfind("type", 0) != 0)
	throw std::runtime_error("Expected 'type ...' header");

	// "height H"
	if (!std::getline(in, line)) return false; rstrip_cr(line);
	const int H = parse_kv_int(line, "height");

	// "width W"
	if (!std::getline(in, line)) return false; rstrip_cr(line);
	const int W = parse_kv_int(line, "width");

	// "map"
	if (!std::getline(in, line)) return false; rstrip_cr(line);
	if (line != "map")
	throw std::runtime_error("Expected 'map' header line");

	// Column-major allocation: free[x][y]
	out.width = W;
	out.height = H;
	out.free.assign(W, std::vector<uint8_t>(H, 0));

	// Read H rows from TOP to BOTTOM; flip Y into math coords.
	int count_obs = 0;
	for (int y_file = 0; y_file < H; ++y_file) 
		{
		if (!std::getline(in, line)) return false;
		rstrip_cr(line);
		if ((int)line.size() != W)
		throw std::runtime_error("Row width mismatch at file row y=" + std::to_string(y_file));
		const int y_math = H - 1 - y_file;

		for (int x = 0; x < W; ++x)
			{
			const uint8_t isfree = is_free_char(line[x]) ? 1u : 0u;
			out.free[x][y_math] = isfree;
			if (!isfree) ++count_obs; 
			}
		}

	out.obs = count_obs;

	return true;
	}

bool loadScenario(const std::string& scenPath, int K, int mapHeight, std::vector<AGENT>& agents_out)
	{
	std::ifstream in(scenPath);
	if (!in) return false;

	agents_out.clear();

	std::string line;
	if (!std::getline(in, line)) return false;
	rstrip_cr(line);
	// Expect "version ..." on first line
	if (line.rfind("version", 0) != 0)
	throw std::runtime_error("Scenario file missing 'version' line");

	int loaded = 0;
	while (std::getline(in, line)) 
		{
		rstrip_cr(line);
		if (line.empty() || line[0] == '#') continue;

		// Fields: bucket map_path width height sx sy gx gy optimal_length
		std::istringstream iss(line);
		std::string bucket, map_path;
		int w=0, h=0, sx_file=0, sy_file=0, gx_file=0, gy_file=0;
		double optlen = 0.0;

		if (!(iss >> bucket >> map_path >> w >> h >> sx_file >> sy_file >> gx_file >> gy_file >> optlen))
			throw std::runtime_error("Bad .scen line: " + line);

		// Convert to math coords.
		auto [sx, sy] = file_to_math(sx_file, sy_file, mapHeight);
		auto [gx, gy] = file_to_math(gx_file, gy_file, mapHeight);

		agents_out.push_back(AGENT{sx, sy, gx, gy});
		++loaded;
		if (K > 0 && loaded >= K) break;
		}

	if (agents_out.empty() && K != 0)
	throw std::runtime_error("No agents loaded from scenario (check K or file).");

	return true;
	}

bool loadInstance(const std::string& mapPath,
		const std::string& scenPath,
		int K,
		GRID& grid_out,
		std::vector<AGENT>& agents_out,
		bool verify_passable)
	{
	if (!loadMap(mapPath, grid_out))
		throw std::runtime_error("Failed to read map: " + mapPath);

	if (!loadScenario(scenPath, K, grid_out.height, agents_out))
		throw std::runtime_error("Failed to read scenario: " + scenPath);

	if (verify_passable) 
		{
		for (size_t i = 0; i < agents_out.size(); ++i) 
			{
			const auto& a = agents_out[i];
			if (!grid_out.inBounds(a.sx, a.sy) || !grid_out.inBounds(a.gx, a.gy))
				throw std::runtime_error("AGENT " + std::to_string(i) + " start/goal out of bounds");

			if (grid_out.at(a.sx, a.sy) == 0u)
				throw std::runtime_error("AGENT " + std::to_string(i) + " start is blocked");

			if (grid_out.at(a.gx, a.gy) == 0u)
			throw std::runtime_error("AGENT " + std::to_string(i) + " goal is blocked");
			}
		}

	return true;
	}

