#include <iostream>
#include <cstddef>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <string>
#include <utility>
#include <random>
#include <limits>
#include <algorithm>
#include <array>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cassert>

#include "ArrayND.hpp"
#include "MapfIO.hpp"

// Structs

struct TaskID 
	{
	int start; // node id
	int goal;  // node id
	};

struct PairHash 
	{
	std::size_t operator()(const std::pair<int,int>& p) const noexcept 
		{
		return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
		}
	};

struct Item 
	{
	double    val;          // sort key
	uint16_t time, agent, node1, node2;       // - If indices < 65,536, otherwise set it to 32.
	};


double eta_pos = 0.01; 


// Global variables
int Total_Cost;
int Zags, Ztime;
int Zobs, Znodes, Zedges;
std::unordered_set<std::pair<int,int>, PairHash> Obstacles;

std::vector<int> coord2id;
std::vector<std::pair<int,int>> id2coord;

std::vector<std::pair<int,int>> edges;

std::vector<TaskID> Tasks;

GRID Map;
std::vector<AGENT> Agents;

Array3D<int> FX;

// RRR variables

Array3D<double> XG, XGA, XGR, XGB;	Array2D<double> etaXG, errXG;	double tXGerr;
Array2D<double> ZC, ZCA, ZCR, ZCB;	Array2D<double> etaZC, errZC;	double tZCerr;

Array3D<double[2]> XC, XCA, XCR, XCB;	Array2D<double> etaXC, errXC;	double tXCerr;
Array3D<double[2]> XE, XEA, XER, XEB;	Array2D<double> etaXE, errXE;	double tXEerr;

double toterr;
double beta,epsilon;
int iter;

// Solution

Array3D<double> XT;
Array3D<double> YT;

// Aggregate Weights

Array3D<double> ZVarX;
Array3D<double> ZVarY;

// Output Files
std::string errfile, statsfile, solfile, SeqFile;


std::mt19937 gen(12345);

void urand_seed(unsigned int s) { gen.seed(s); }

inline double sq(double diff) { return diff * diff; }
inline int idx(int i, int j) { return i * Map.height + j; }

inline void sort_items(std::vector<Item>& a)
	{
	std::sort(a.begin(), a.end(), [](const Item& x, const Item& y) {return x.val < y.val;}  );   // ascending
	}

double urand(double a, double b)
	{
	std::uniform_real_distribution<double> dist(a, b);
	return dist(gen);
	}

struct Map 
	{
	int width = 0, height = 0;

	std::vector<std::vector<uint8_t>> free;      // free[y][x] in {0,1}

	// Convenience accessor
	inline uint8_t at(int x, int y) const { return free[y][x]; }
	};

bool setupVariables()
	{
	Znodes = Map.height * Map.width - Map.obs;

	coord2id.assign(Map.width * Map.height, -1);
	id2coord.clear();
	id2coord.reserve(Znodes);

	int id = 0;
	for (int i = 0; i < Map.width; ++i)
		for (int j = 0; j < Map.height; ++j)
			if (Map.free[i][j]) 
				{
				coord2id[idx(i,j)] = id;
				id2coord.push_back({i, j});
				++id;
				}

	// Edges

	//static constexpr std::array<std::pair<int,int>, 5> dirs = {{ {0, 0}, {0, 1}, {1, 0}, {0,-1}, {-1,0} }};
	static constexpr std::array<std::pair<int,int>, 5> dirs = {{ {0, 1}, {1, 0}, {0,-1}, {-1,0} }}; 	// No waiting

	edges.clear();
	edges.reserve(static_cast<size_t>(Znodes) * 4);

	Zedges = 0;
	for(int i = 0; i < Znodes; ++i)
		{
		auto [x, y] = id2coord[i];
		for (auto [dx, dy] : dirs) 
			{
			int x2 = x + dx, y2 = y + dy;

			// in-bounds?
			if (x2 < 0 || x2 >= Map.width || y2 < 0 || y2 >= Map.height)
				continue;

			int j = coord2id[idx(x2,y2)];
			if (j < 0) // obstacle?
				continue;
			
			if (i > j)
				continue;
		
			else
				{
				edges.emplace_back(i, j);
				edges.emplace_back(j, i);
				Zedges += 2;
				}
			}
		}

	if (Zedges != static_cast<int>(edges.size()))
		{
		std::cerr << "Error: edge book keeping out of sync\n";
		return 0;
		}

	std::ofstream stats(statsfile, std::ios::app);

	stats   << "Number of agents: "     << Zags   << '\n'
		<< "Number of grid sites: " << Znodes << '\n'
		<< "Number of edges: "      << Zedges << '\n'
		<< "Number of obstacles: "  << Map.obs<< '\n'
		<< "Alloted time: "         << Ztime  << "\n\n";

	Tasks.clear();
	Tasks.reserve(Zags);

	for (int i = 0; i < Zags; ++i) 
		{
		Tasks.push_back({coord2id[idx(Agents[i].sx, Agents[i].sy)], coord2id[idx(Agents[i].gx, Agents[i].gy)]});
		stats << Agents[i].sx << '\t' << Agents[i].sy << "\t --> \t" << Agents[i].gx << '\t' << Agents[i].gy << '\n'; 
		}


	stats.close();

	
	FX.alloc(Ztime+1,Zags,Znodes, true);

	for(int t = 0; t < Ztime+1; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				{
				auto [x, y] = id2coord[n];
				const int dis_start = std::abs(x-Agents[a].sx) + std::abs(y-Agents[a].sy); 
				const int dis_end = std::abs(x-Agents[a].gx) + std::abs(y-Agents[a].gy);

				if (dis_start <= t && dis_end <= Ztime-t)
					{
					if (n != Tasks[a].goal && ((t - dis_start) & 1) == 0)
						FX[t][a][n] = 1;
					else if (n == Tasks[a].goal)
						FX[t][a][n] = 1;
					}
				}

	return true;
	}


void makevars()
	{
	// RRR iterates

	ZC.alloc(Ztime, Zags, true);
	ZCA.alloc(Ztime, Zags, false);
	ZCR.alloc(Ztime, Zags, false);
	ZCB.alloc(Ztime, Zags, false);

	XC.alloc(Ztime, Zags, Znodes+2, true);
	XCA.alloc(Ztime, Zags, Znodes+2, false);
	XCR.alloc(Ztime, Zags, Znodes+2, false);
	XCB.alloc(Ztime, Zags, Znodes+2, false);

	XG.alloc(Ztime+1, Znodes, Zags, true);
	XGA.alloc(Ztime+1, Znodes, Zags, false);
	XGR.alloc(Ztime+1, Znodes, Zags, false);
	XGB.alloc(Ztime+1, Znodes, Zags, false);

	XE.alloc(Ztime, Zedges, Zags, true);
	XEA.alloc(Ztime, Zedges, Zags, false);
	XER.alloc(Ztime, Zedges, Zags, false);
	XEB.alloc(Ztime, Zedges, Zags, false);

	// Metric parameters
	etaXC.alloc(Ztime, Zags, false);
	errXC.alloc(Ztime, Zags, false);

	etaZC.alloc(Ztime, Zags, false);
	errZC.alloc(Ztime, Zags, false);

	etaXG.alloc(Ztime+1, Znodes, false);
	errXG.alloc(Ztime+1, Znodes, false);

	etaXE.alloc(Ztime, Zedges, false);
	errXE.alloc(Ztime, Zedges, false);

	// Other parameters

	XT.alloc(Ztime+1, Zags, Znodes + 2, false);
	ZVarX.alloc(Ztime+1, Zags, Znodes + 2, false);
	}

void changeVar(Array3D<double[2]> &XCo, Array3D<double> &XGo, Array3D<double[2]> &XEo)
	{
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes + 2; ++n)
				{
				XCo[t][a][n][0] = XT[t][a][n];
				XCo[t][a][n][1] = XT[t+1][a][n];
				}
		
	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			for(int a = 0; a < Zags; ++a)
				XGo[t][n][a] = XT[t][a][n];

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			for(int a = 0; a < Zags; ++a)
				{
				auto [n1,n2] = edges[e];
				XEo[t][e][a][0] = XT[t][a][n1];
				XEo[t][e][a][1] = XT[t+1][a][n2];
				}
	}


void projA(const Array3D<double[2]> &XCo, const Array3D<double> &XGo, const Array3D<double[2]> &XEo, const Array2D<double> &ZCo)
	{
	// Connectivity constraint constraint + existence

	std::vector<Item> items(Ztime * Zags);

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			double min_cost = std::numeric_limits<double>::infinity();
			int proj_index = -1;

			for(int n = 0; n < Znodes; ++n)
				{
				XCA[t][a][n][0] = 0.;
				XCA[t][a][n][1] = 0.;
				}

			for(int e = 0; e < Zedges; ++e)
				{
				auto [n1,n2] = edges[e];

				if (FX[t][a][n1] == 0 || FX[t+1][a][n2] == 0)
					continue;

				auto [x1,y1] = id2coord[n1];
				auto [x2,y2] = id2coord[n2];

				double cur_cost = 2. * (1. - XCo[t][a][n1][0] - XCo[t][a][n2][1]) 
						+ eta_pos * ( sq(x1 - XCo[t][a][Znodes][0]) + sq(y1 - XCo[t][a][Znodes+1][0])
							    + sq(x2 - XCo[t][a][Znodes][1]) + sq(y2 - XCo[t][a][Znodes+1][1])); 

				if (cur_cost < min_cost)
					{
					min_cost = cur_cost;
					proj_index = e;
					}
				}

			if (FX[t][a][Tasks[a].goal] == 0)
				{
				auto [n1,n2] = edges[proj_index];
				XCA[t][a][n1][0] = 1.;
				XCA[t][a][n2][1] = 1.;

				auto [x1,y1] = id2coord[n1];
				auto [x2,y2] = id2coord[n2];

				XCA[t][a][Znodes][0] = x1;
				XCA[t][a][Znodes+1][0] = y1;
				XCA[t][a][Znodes][1] = x2;
				XCA[t][a][Znodes+1][1] = y2;
	
				ZCA[t][a] = 1.;
				}

			else
				{
				min_cost += (1. - 2. * ZCo[t][a]);

				auto [xg,yg] = id2coord[Tasks[a].goal];

				double self_cost = 2. * (1. - XCo[t][a][Tasks[a].goal][0] - XCo[t][a][Tasks[a].goal][1]) 
						    + eta_pos * ( sq(xg - XCo[t][a][Znodes][0]) + sq(yg - XCo[t][a][Znodes+1][0])
								+ sq(xg - XCo[t][a][Znodes][1]) + sq(yg - XCo[t][a][Znodes+1][1])); 

				if (self_cost < min_cost)
					{
					XCA[t][a][Tasks[a].goal][0] = 1.;
					XCA[t][a][Tasks[a].goal][1] = 1.;
			
					ZCA[t][a] = 0.;

					auto [xg,yg] = id2coord[Tasks[a].goal];
					XCA[t][a][Znodes][0] = xg;
					XCA[t][a][Znodes+1][0] = yg;
					XCA[t][a][Znodes][1] = xg;
					XCA[t][a][Znodes+1][1] = yg;
					}

				else
					{
					auto [n1,n2] = edges[proj_index];
					XCA[t][a][n1][0] = 1.;
					XCA[t][a][n2][1] = 1.;

					auto [x1,y1] = id2coord[n1];
					auto [x2,y2] = id2coord[n2];

					XCA[t][a][Znodes][0] = x1;
					XCA[t][a][Znodes+1][0] = y1;
					XCA[t][a][Znodes][1] = x2;
					XCA[t][a][Znodes+1][1] = y2;
		
					ZCA[t][a] = 1.;
					}
				}
			}



/*
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			double max_inverse_cost = (FX[t][a][Tasks[a].goal] == 0) ? -std::numeric_limits<double>::infinity() : XCo[t][a][Tasks[a].goal][0] + XCo[t][a][Tasks[a].goal][1];
			int proj_index = -1;

			for(int n = 0; n < Znodes; ++n)
				{
				XCA[t][a][n][0] = 0.;
				XCA[t][a][n][1] = 0.;
				}

			for(int e = 0; e < Zedges; ++e)
				{
				auto [n1,n2] = edges[e];

				if (FX[t][a][n1] == 0 || FX[t+1][a][n2] == 0)
					continue;

				double cur_inverse_cost = XCo[t][a][n1][0] + XCo[t][a][n2][1]; 

				if (cur_inverse_cost > max_inverse_cost)
					{
					max_inverse_cost = cur_inverse_cost;
					proj_index = e;
					}
				}

			if (proj_index == -1)
				{
				XCA[t][a][Tasks[a].goal][0] = 1.;
				XCA[t][a][Tasks[a].goal][1] = 1.;
				}
			else
				{
				auto [n1,n2] = edges[proj_index];
				XCA[t][a][n1][0] = 1.;
				XCA[t][a][n2][1] = 1.;
				}
			}
*/

	// Each node can have atmost one agent (conflict-resolution)

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			{
			if (Zags == 0) continue;

			double*		XGA_tn = XGA[t][n];
			const double*	XGo_tn = XGo[t][n];

			XGA.fill_row(t, n, 0.0);

			// pick agent with maximum XGo value (since 1 - 2x is minimized when x is max). This checks only for 1-hot vectors
			const double* it = std::max_element(XGo_tn, XGo_tn + Zags);
			const int proj_index = static_cast<int>(it - XGo_tn);

			if (XGo_tn[proj_index] > 0.5)
				XGA_tn[proj_index] = 1.0;
			// else it should be a zero vector
			}

/*
	// Defined for each edge that includes its complementary edge: For each agent and time instant, YE should be a one-hot vector or zero-vector (edges are travelled by atmost 1 agent) 
	// Size of Vector: Number of Agents x2
	// Each undirected edge can only have at most 1 agent

	for(int t = 0; t < Ztime; ++t)
		{
		int e = 0;
		while (e < Zedges)
			{
			auto [n1,n2] = edges[e];

			if (Zags == 0) continue;

			if (n1 < n2)
				{
				assert(e + 1 < Zedges);
				// combined agent vector for the (n1,n2) and (n2,n1) edges should be either zero or 1-hot vector 

				double*		YEA_te1 = YEA[t][e];
				const double*	YEo_te1 = YEo[t][e];

				YEA.fill_row(t, e, 0.0);
				
				const double* it1 = std::max_element(YEo_te1, YEo_te1 + Zags);
				const int min_index1 = static_cast<int>(it1 - YEo_te1);

				double*		YEA_te2 = YEA[t][e+1];
				const double*	YEo_te2 = YEo[t][e+1];

				YEA.fill_row(t, e+1, 0.0);

				const double* it2 = std::max_element(YEo_te2, YEo_te2 + Zags);
				const int min_index2 = static_cast<int>(it2 - YEo_te2);

				if (YEo_te1[min_index1] > 0.5 && YEo_te1[min_index1] > YEo_te2[min_index2] )
					YEA_te1[min_index1] = 1.0;

				else if (YEo_te2[min_index2] > 0.5 && YEo_te2[min_index2] > YEo_te1[min_index1] )
					YEA_te2[min_index2] = 1.0;

				e += 2;
				}

			else 
				e++;
			}
		}
*/
	}


void reflect(const Array3D<double[2]> &XCo, const Array3D<double> &XGo, const Array3D<double[2]> &XEo, const Array2D<double> &ZCo)
	{
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes + 2; ++n)
				{
				XCR[t][a][n][0] = 2.*XCo[t][a][n][0] - XC[t][a][n][0];
				XCR[t][a][n][1] = 2.*XCo[t][a][n][1] - XC[t][a][n][1];
				}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			ZCR[t][a] = 2.*ZCo[t][a] - ZC[t][a];

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			for(int a = 0; a < Zags; ++a)
				XGR[t][n][a] = 2.*XGo[t][n][a] - XG[t][n][a];

/*
	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			for(int a = 0; a < Zags; ++a)
				{
				XER[t][e][a][0] = 2.*XEo[t][e][a][0] - XE[t][e][a][0];
				XER[t][e][a][1] = 2.*XEo[t][e][a][1] - XE[t][e][a][1];
				}
*/
	}

void projB(const Array3D<double[2]> &XCo, const Array3D<double> &XGo, const Array3D<double[2]> &XEo, const Array2D<double> &ZCo)
	{
	for(int t = 0; t < Ztime+1; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			std::fill_n(XT[t][a],    Znodes + 2, 0.0);
			std::fill_n(ZVarX[t][a], Znodes + 2, 0.0);
			}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			double eta = etaXC[t][a]; 
			for(int n = 0; n < Znodes + 2; ++n)
				{
				XT[t][a][n] += eta * XCo[t][a][n][0];
				ZVarX[t][a][n] += eta;

				XT[t+1][a][n] += eta * XCo[t][a][n][1];
				ZVarX[t+1][a][n] += eta;
				}
			}

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			{
			double eta = etaXG[t][n]; 

			for(int a = 0; a < Zags; ++a)
				{
				XT[t][a][n] += eta * XGo[t][n][a];
				ZVarX[t][a][n] += eta;
				}
			}

/*
	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; e += 2)
			{
			double eta = etaXE[t][e]; 

			for(int a = 0; a < Zags; ++a)
				{
				auto [n1,n2] = edges[e];
				XT[t][a][n1] += eta * XEo[t][e][a][0];
				XT[t+1][a][n2] += eta * XEo[t][e][a][1];

				XT[t][a][n2] += eta * XEo[t][e+1][a][0];
				XT[t+1][a][n1] += eta * XEo[t][e+1][a][1];
				}
			}
*/

	for(int t = 0; t < Ztime+1; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes + 2; ++n)
				XT[t][a][n] /= ZVarX[t][a][n];

	// Reducing the total cost

	double current_cost = 0.;
	double lambda = 0;

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			ZCB[t][a] = ZCo[t][a];
			current_cost += ZCo[t][a];
			lambda += 1./etaZC[t][a]; 
			}

	if (current_cost > Total_Cost)
		for(int t = 0; t < Ztime; ++t)
			for(int a = 0; a < Zags; ++a)
				ZCB[t][a] -= (current_cost - Total_Cost)/(etaZC[t][a]*lambda);

	changeVar(XCB, XGB, XEB);
	}

void RRR()
	{    
	int totVar, totCons;
	double diff, avgerr;

	projA(XC, XG, XE, ZC);
	reflect(XCA, XGA, XEA, ZCA);
	projB(XCR, XGR, XER, ZCR);

	tXGerr = 0.;
	tXEerr = 0.;
	tXCerr = 0.;
	tZCerr = 0.;
	toterr = 0.;
	avgerr = 0.;

	totVar = 0; 
	totCons = 0; 


	// Consistency Constraint between X and Y variables 

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			errZC[t][a] = 0.;

			diff = ZCB[t][a] - ZCA[t][a];
			ZC[t][a] += beta*diff;
			errZC[t][a] += sq(diff);
			tZCerr += errZC[t][a];
			avgerr += errZC[t][a];


			errXC[t][a] = 0.;

			for(int n = 0; n < Znodes; ++n)
				{
				diff = XCB[t][a][n][0] - XCA[t][a][n][0];
				XC[t][a][n][0] += beta*diff;
				errXC[t][a] += sq(diff);

				diff = XCB[t][a][n][1] - XCA[t][a][n][1];
				XC[t][a][n][1] += beta*diff;
				errXC[t][a] += sq(diff);
				}

			for(int n = Znodes; n < Znodes + 2; ++n)
				{
				diff = XCB[t][a][n][0] - XCA[t][a][n][0];
				XC[t][a][n][0] += beta*diff;
				errXC[t][a] += eta_pos * sq(diff);

				diff = XCB[t][a][n][1] - XCA[t][a][n][1];
				XC[t][a][n][1] += beta*diff;
				errXC[t][a] += eta_pos * sq(diff);
				}

			tXCerr += errXC[t][a];
			errXC[t][a] /= 2.*(Znodes + 2);

			avgerr += errXC[t][a];
			}

        totVar += Ztime * Zags;
        toterr += tZCerr;
        tZCerr /= Ztime * Zags;
        totCons += Ztime * Zags;

        totVar += Ztime * Zags * 2. * (Znodes + 2.);
        toterr += tXCerr;

        tXCerr /= Ztime * Zags * 2. * (Znodes + 2.);
        totCons += Ztime * Zags;

	// Conflict resolution at a node

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			{
			errXG[t][n] = 0.;

			for(int a = 0; a < Zags; ++a)
				{
				diff = XGB[t][n][a] - XGA[t][n][a];
				XG[t][n][a] += beta*diff;
				errXG[t][n] += sq(diff);
				}

			tXGerr += errXG[t][n];
			errXG[t][n] /= Zags;
			avgerr += errXG[t][n];
			}

        totVar += (Ztime+1)*Zags*Znodes;
        toterr += tXGerr;

        tXGerr /= (Ztime+1)*Zags*Znodes;
        totCons += (Ztime+1)*Znodes;

/*
	// No crossing conflict resolution 

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; e += 2)
			{
			errXE[t][e] = 0.;
			
			auto [n1,n2] = edges[e];

			assert(e & 1 == 0);  // e must be even

			totCons++;

			for(int a = 0; a < Zags; ++a)
				for(int c = 0; c < 2; ++c)
					{
					diff = XEB[t][e][a][c] - XEA[t][e][a][c];
					XE[t][e][a][c] += beta*diff;
					errXE[t][e] += sq(diff);

					diff = XEB[t][e+1][a][c] - XEA[t][e+1][a][c];
					XE[t][e+1][a][c] += beta*diff;
					errXE[t][e] += sq(diff);
					}

			tXEerr += errXE[t][e];
			errXE[t][e] /= Zags*4.;
			avgerr += errXE[t][e];
			}

        totVar += Ztime * Zags * Zedges * 4;
        toterr += tXEerr;

        tXEerr /= Ztime * Zags * Zedges * 4;
*/
			
	toterr /= totVar;
	avgerr /= totCons;

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			etaZC[t][a] += epsilon*(errZC[t][a] - etaZC[t][a] * avgerr);

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			etaXC[t][a] += epsilon*(errXC[t][a] - etaXC[t][a] * avgerr);
			//etaXC[t][a] += epsilon*(errXC[t][a]/avgerr - etaXC[t][a]);

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			etaXG[t][n] += epsilon*(errXG[t][n] - etaXG[t][n] * avgerr);
			//etaXG[t][n] += epsilon*(errXG[t][n]/avgerr - etaXG[t][n]);

/*
	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; e+=2)
			etaXE[t][e] += epsilon*(errXE[t][e] - etaXE[t][e] * avgerr);
*/

	tXGerr = sqrt(tXGerr);
	tXEerr = sqrt(tXEerr);
	tXCerr = sqrt(tXCerr);
	tZCerr = sqrt(tZCerr);

	toterr = sqrt(toterr);

        }


void printsol(const std::string& filename) 
	{

	std::vector<std::vector<std::array<int,3>>> agent_pos(Zags, std::vector<std::array<int,3>>(Ztime + 1));

	for(int a = 0; a < Zags; ++a)
		for(int t = 0; t < Ztime+1; ++t)
			{
			const double* XT_ta = XT[t][a];

			const double* it = std::max_element(XT_ta, XT_ta + Znodes);
			const int n = static_cast<int>(it - XT_ta);

			auto [r,c] = id2coord[n];
			agent_pos[a][t] = {t, r, c}; 
			}

	std::ofstream ofs(filename, std::ios::app);

	if (!ofs) 
		{
		std::cerr << "Error opening file " << filename << "\n";
		return;
		}

	//Dump positions: one line per agent
	for (int a = 0; a < Zags; ++a) 
		{
		for (int t = 0; t < Ztime + 1; ++t) 
			{
			const auto& tac = agent_pos[a][t];

			//if (tac[1] == -1)
			//	continue;

			//ofs << tac[0] << ','<< tac[1] << ',' << tac[2];
			ofs << tac[1] << ',' << tac[2];
			if (t < Ztime) ofs << ';';  // separator between timesteps
			}
		ofs << '\n'; // newline = next agent
		}
	}


int solve(int maxiter, int iterstride, double stoperr, bool log_iterations) 
	{
	std::ofstream fperr(errfile, std::ios::trunc);   // overwrite (like "w")

	if (!fperr) 
		{
		std::cerr << "Error opening error log file: " << errfile << "\n";
		return -1;
		}

	if (iterstride <= 0) 
		{
		std::cerr << "iterstride must be positive\n";
		return -1;
		}
	

	fperr.setf(std::ios::scientific);
	fperr << std::setprecision(6);

	for (iter = 1; iter <= maxiter; ++iter) 
		{
		RRR();

		if (log_iterations)
			{
			const bool log_now = (iter == 1) || (iter % iterstride == 0) || (toterr < stoperr);

			if (log_now) 
				{
				fperr << iter << '\t'
				 << toterr << '\t'
				 << tXCerr << ' '
				 << tXGerr << ' '
				 << tXEerr << ' '
				 << '\t'<< tZCerr << ' '
				 //<< '\t'<< TYerr << ' '
				 //<< '\t'<< etaX << ' '
				 //<< '\t'<< etaY << ' '
				 << '\n';

				printsol(solfile);
				}
			}

/*
		if (iter == maxiter)
			{
			for(int t = 0; t < Ztime; ++t)
				{
				for(int a = 0; a < Zags; ++a)
					{
					printf("%.4e \t %lf \t\n", errXY[t][a], etaXY[t][a]);
	
					if (errXY[t][a] > 1e-10)
						{
						for(int n = 0; n < Znodes; ++n)
							for(int c = 0; c < 2; ++c)
								{	
								double diff = XCB[t][a][n][c] - XCA[t][a][n][c];
								if (sq(diff) > 1e-20)
									printf("%d %d %d %d \t %lf \t %lf\n", t,a,n,c, XCB[t][a][n][c] , XCA[t][a][n][c]);
								}

						for(int e = 0; e < Zedges; ++e)
							{
							double diff = YCB[t][a][e] - YCA[t][a][e];
							if (sq(diff) > 1e-20)
								printf("%d %d %d \t %lf \t %lf\n", t,a,e, YCB[t][a][e] , YCA[t][a][e]);
							}
						}
					}
				printf("\n");
				}
			}
*/

		if (toterr < stoperr) 
			return iter;

		}
	
	return 0;
	}

void initialize_metric_parameters()
	{

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			etaXC[t][a] = 1.;
			etaZC[t][a] = 1.;
			}

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			etaXG[t][n] = 1.;

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			etaXE[t][e] = 1.;
	}

void init()
	{
	for(int t = 1; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				XT[t][a][n] = urand(0.0,1.0);

	for(int t = 0; t <= Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			XT[t][a][Znodes] = urand(0.0,Map.width);
			XT[t][a][Znodes+1] = urand(0.0,Map.height);
			}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			ZC[t][a] = urand(0.0,1.0);
			
				
	for(int a = 0; a < Zags; ++a)
		for(int n = 0; n < Znodes; ++n)
			{
			XT[0][a][n] = 0.0;
			XT[Ztime][a][n] = 0.0;
			}

	for(int a = 0; a < Zags; ++a)
		{
		XT[0][a][Tasks[a].start] = 1.0;
		XT[Ztime][a][Tasks[a].goal] = 1.0;
		}

	changeVar(XC, XG, XE);

	initialize_metric_parameters();
	}

void initSol(const std::string& filename, double lambda) 
	{
	std::ifstream ifs(filename);
	if (!ifs) 
		{
		std::cerr << "Error opening file " << filename << "\n";
		return;
		}

	// Zero out XT, YT first
	for (int t = 0; t < Ztime + 1; ++t) 
		for (int a = 0; a < Zags; ++a) 
			std::fill_n(XT[t][a], Znodes, 0.0);

	for (int t = 0; t < Ztime; ++t) 
		for (int a = 0; a < Zags; ++a) 
			ZC[t][a] = 1.0;

	std::string line;
	int agent_id = 0;
	while (std::getline(ifs, line) && agent_id < Zags) 
		{
		std::stringstream ss(line);
		std::string position;

		int t = 0;
		while (std::getline(ss, position, ';')) 
			{
			std::stringstream ts(position);
			std::string val;
			int x, y;

			std::getline(ts, val, ','); x = std::stoi(val);
			std::getline(ts, val, ','); y = std::stoi(val);

			int n = coord2id[idx(x,y)];
			if (n >= 0 && n < Znodes) 
				XT[t][agent_id][n] = 1.0;

			XT[t][agent_id][Znodes] = x;
			XT[t][agent_id][Znodes+1] = y;

			if (t < Ztime && n == Tasks[agent_id].goal)
				ZC[t][agent_id] = 0.0;
			t++;
			}

		++agent_id; // next agent line
		}

	for (int t = 0; t < Ztime; ++t) 
		for (int a = 0; a < Zags; ++a) 
			ZC[t][a] += lambda * urand(-1.0,1.0);


	for(int t = 1; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				XT[t][a][n] += lambda * urand(-1.0,1.0);

	changeVar(XC, XG, XE);

	initialize_metric_parameters();
	}

void run_experiments(int numrun, int maxiter, int iterstride, double stoperr, int seed)
	{
	std::ios::sync_with_stdio(false);

	std::ofstream stat(statsfile, std::ios::app);

	stat << std::fixed << std::setprecision(2);		

	int totiter_success = 0;
	int tot_success = 0;

	bool write_each_run = (numrun == 1);

	for (int i = 0; i < numrun; ++i)
		{
		urand_seed(seed+i); 
		//initSol("solution.txt", 0.5);
		init();

		auto t0 = std::chrono::high_resolution_clock::now();
		int iter_needed = solve(maxiter, iterstride, stoperr, write_each_run);
		auto t1 = std::chrono::high_resolution_clock::now();

		// Elapsed seconds as double
		double deltat = std::chrono::duration<double>(t1 - t0).count();

		if (iter_needed)  // success path
			{
			totiter_success += iter_needed;
			++tot_success;
			double iterpersec = iter_needed / deltat;
			stat << iter_needed << '\t' << deltat << '\t' << iterpersec << '\n';
			stat.flush();
			printf("%d \t %d \n",i,iter_needed);
			fflush(stdout);
			}
		else       // failure path: write -1 as in C code
			{
			double iterpersec = maxiter / deltat;
			stat << -1  << '\t' << deltat << '\t' << iterpersec << '\n';
			stat.flush();
			}

		}

	if (numrun != 1)
		{
		if (tot_success > 0)
			std::cout << "\n Avg iter needed:" << std::fixed << std::setprecision(6) << (1.0 * totiter_success) / tot_success << "\n";
		else
			std::cout << "Avg iter needed: n/a (no successes)\n";

		std::cout << "Total successes:" << tot_success << "\n";
		}
	}


int main(int argc, char* argv[]) 
	{
	if (argc != 14) 
		{
		std::cerr
		<< "Usage:\n  " << argv[0]
		<< " <mapPath> <scenarioPath> <Zags> <runID> <Ztime> <Total_Cost> <beta> <maxiter> <iterstride> <stoperr> <epsilon> <seed> <numrun>\n";
		return EXIT_FAILURE;
		}

	// Parse args 
	const std::string mapPath  	= argv[1];
	const std::string scenarioPath 	= argv[2];
	Zags	 			= std::stoi(argv[3]);
	const std::string runID        	= argv[4];
	Ztime				= std::stoi(argv[5]);
	Total_Cost			= std::stoi(argv[6]);
	beta  			    	= std::stod(argv[7]);
	const int maxiter		= std::stoi(argv[8]);
	const int iterstride		= std::stoi(argv[9]);
	const double stoperr   		= std::stod(argv[10]);
	epsilon 			= std::stod(argv[11]);
	const int seed		      	= std::stoi(argv[12]);
	const int numrun		= std::stoi(argv[13]);

	// Derived filenames 
	errfile   = runID    + ".err";
	statsfile = runID    + ".stats";
	solfile   = runID    + ".sol";

	std::ofstream ofs(solfile, std::ios::trunc);
	if (!ofs) 
		{
		std::cerr << "Error creating " << solfile << "\n";
		return 1;
		}
	ofs.close();

	std::ofstream stats(statsfile, std::ios::trunc);
	if (!stats) 
		{
		std::cerr << "Error creating " << statsfile << "\n";
		return 1;
		}

	// Write all argv[c] separated by space
	for (int c = 0; c < argc; ++c) 
		stats << argv[c] << ' ';
	stats << "\n\n";

	try 
		{
		loadInstance(mapPath, scenarioPath, Zags, Map, Agents, /*verify_passable=*/true);
		} 
	catch (const std::exception& e) 
		{
		std::cerr << "Load error: " << e.what() << "\n";
		return 2;
		}

	stats << "Loaded map " << Map.width << "x" << Map.height << " with " << Agents.size() << " agents.\n";
	stats.close();

	setupVariables();

	makevars();

	run_experiments(numrun, maxiter, iterstride, stoperr, seed);

	return 0;
	}
