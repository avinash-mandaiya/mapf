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
	uint16_t i, j, k;       // - If indices < 65,536, otherwise set it to 32.
	};

double TXerr, TYerr;

// Global variables
int Zags, Ztime;
int Zobs, Znodes, Zedges;
std::unordered_set<std::pair<int,int>, PairHash> Obstacles;

std::vector<int> coord2id;
std::vector<std::pair<int,int>> id2coord;

std::vector<std::pair<int,int>> edges;
Array2D<int> node2edge;

std::vector<int> self_edge_index;
std::vector<TaskID> Tasks;

GRID Map;
std::vector<AGENT> Agents;

Array3D<int> FX;

// RRR variables

Array3D<double> XG, XGA, XGR, XGB;	Array2D<double> etaXG, errXG;	double tXGerr;
Array3D<double> YE, YEA, YER, YEB;	Array2D<double> etaYE, errYE;	double tYEerr;

Array3D<double> YC, YCA, YCR, YCB; 	Array2D<double> etaXY, errXY; 	double tXYerr;
Array3D<double[2]> XC, XCA, XCR, XCB;

double toterr;
double beta,epsilon;
int iter;

// Tuning parameters
double etaX, etaY;

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

	node2edge.alloc(Znodes, Znodes, false);

	for(int i = 0; i < Znodes; ++i)
		for(int j = 0; j < Znodes; ++j)
			node2edge[i][j] = -1;

	static constexpr std::array<std::pair<int,int>, 5> dirs = {{ {0, 0}, {0, 1}, {1, 0}, {0,-1}, {-1,0} }};

	self_edge_index.assign(Znodes, -1);

	edges.clear();
	edges.reserve(static_cast<size_t>(Znodes) * 5);

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
			
			if (i == j)
				{
				edges.emplace_back(i, i);
				self_edge_index[i] = Zedges;
				node2edge[i][i] = Zedges;
				Zedges++;
				}

			else if (i > j)
				continue;
		
			else
				{
				edges.emplace_back(i, j);
				node2edge[i][j] = Zedges;
				edges.emplace_back(j, i);
				node2edge[j][i] = Zedges+1;
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

	XC.alloc(Ztime, Zags, Znodes, true);
	XCA.alloc(Ztime, Zags, Znodes, false);
	XCR.alloc(Ztime, Zags, Znodes, false);
	XCB.alloc(Ztime, Zags, Znodes, false);

	YC.alloc(Ztime, Zags, Zedges, true);
	YCA.alloc(Ztime, Zags, Zedges, false);
	YCR.alloc(Ztime, Zags, Zedges, false);
	YCB.alloc(Ztime, Zags, Zedges, false);

	XG.alloc(Ztime+1, Znodes, Zags, true);
	XGA.alloc(Ztime+1, Znodes, Zags, false);
	XGR.alloc(Ztime+1, Znodes, Zags, false);
	XGB.alloc(Ztime+1, Znodes, Zags, false);

	YE.alloc(Ztime, Zedges, Zags, true);
	YEA.alloc(Ztime, Zedges, Zags, false);
	YER.alloc(Ztime, Zedges, Zags, false);
	YEB.alloc(Ztime, Zedges, Zags, false);

	// Metric parameters
	etaXY.alloc(Ztime, Zags, false);
	etaXG.alloc(Ztime+1, Znodes, false);
	etaYE.alloc(Ztime, Zedges, false);

	errXY.alloc(Ztime, Zags, false);
	errXG.alloc(Ztime+1, Znodes, false);
	errYE.alloc(Ztime, Zedges, false);

	// Other parameters

	XT.alloc(Ztime+1, Zags, Znodes, false);
	YT.alloc(Ztime, Zags, Zedges, false);

	ZVarX.alloc(Ztime+1, Zags, Znodes, false);
	ZVarY.alloc(Ztime, Zags, Zedges, false);
	}

void changeVar(Array3D<double[2]> &XCo, Array3D<double> &YCo, Array3D<double> &XGo, Array3D<double> &YEo)
	{
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			for(int e = 0; e < Zedges; ++e)
				YCo[t][a][e] = YT[t][a][e];

			for(int n = 0; n < Znodes; ++n)
				{
				XCo[t][a][n][0] = XT[t][a][n];
				XCo[t][a][n][1] = XT[t+1][a][n];
				}
			}
		
	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			for(int a = 0; a < Zags; ++a)
				XGo[t][n][a] = XT[t][a][n];

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			for(int a = 0; a < Zags; ++a)
				YEo[t][e][a] = YT[t][a][e];
	}


void projA(const Array3D<double[2]> &XCo, const Array3D<double> &YCo, const Array3D<double> &XGo, const Array3D<double> &YEo)
	{
	// Causality constraint + existence
	// YCo = XCo[t] and XCo[t+1]			

	std::vector<Item> cost;
	cost.resize(Ztime * Zags);

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			double min_relative_cost = std::numeric_limits<double>::infinity();
			int min_index = -1;

			for(int e = 0; e < Zedges; ++e)
				YCA[t][a][e] = 0.;

			for(int n = 0; n < Znodes; ++n)
				{
				XCA[t][a][n][0] = 0.;
				XCA[t][a][n][1] = 0.;
				}

			for(int e = 0; e < Zedges; ++e)
				{
				auto [n1,n2] = edges[e];

				if (n1 == n2 && n1 != Tasks[a].goal) // No waiting
					continue;

				if (FX[t][a][n1] == 0 || FX[t+1][a][n2] == 0)
					continue;

				double cur_relative_cost = etaX * 2. * (1. -  XCo[t][a][n1][0] - XCo[t][a][n2][1]) + etaY * (1. - 2. * YCo[t][a][e]);

				if (cur_relative_cost < min_relative_cost)
					{
					min_relative_cost = cur_relative_cost;
					min_index = e;
					}
				}

			YCA[t][a][min_index] = 1.;
			auto [n1,n2] = edges[min_index];
			XCA[t][a][n1][0] = 1.;
			XCA[t][a][n2][1] = 1.;
			}


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
			const int min_index = static_cast<int>(it - XGo_tn);

			if (XGo_tn[min_index] > 0.5)
				XGA_tn[min_index] = 1.0;
			// else it should be a zero vector
			}

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

	// self edge variables in the crossing constraint are unused, so using them to apply the termination constraint


	// No waiting 

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				YEA[t][self_edge_index[n]][a] = 0.;
		

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			int node_index = self_edge_index[Tasks[a].goal];
			if (YEo[t][node_index][a] > 0.5)
				YEA[t][node_index][a] = 1.;
			}
	}


void reflect(const Array3D<double[2]> &XCo, const Array3D<double> &YCo, const Array3D<double> &XGo, const Array3D<double> &YEo)
	{
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			for(int e = 0; e < Zedges; ++e)
				YCR[t][a][e] = 2.*YCo[t][a][e] - YC[t][a][e];

			for(int n = 0; n < Znodes; ++n)
				{
				XCR[t][a][n][0] = 2.*XCo[t][a][n][0] - XC[t][a][n][0];
				XCR[t][a][n][1] = 2.*XCo[t][a][n][1] - XC[t][a][n][1];
				}
			}

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			for(int a = 0; a < Zags; ++a)
				XGR[t][n][a] = 2.*XGo[t][n][a] - XG[t][n][a];

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			for(int a = 0; a < Zags; ++a)
				YER[t][e][a] = 2.*YEo[t][e][a] - YE[t][e][a];
	}

void projB(const Array3D<double[2]> &XCo, const Array3D<double> &YCo, const Array3D<double> &XGo, const Array3D<double> &YEo)
	{
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			std::fill_n(YT[t][a],    Zedges, 0.0);
			std::fill_n(ZVarY[t][a], Zedges, 0.0);
			}

	for(int t = 0; t < Ztime+1; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			std::fill_n(XT[t][a],    Znodes, 0.0);
			std::fill_n(ZVarX[t][a], Znodes, 0.0);
			}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			double eta = etaXY[t][a]; 

			for(int e = 0; e < Zedges; ++e)
				{
				YT[t][a][e] += eta * YCo[t][a][e];
				ZVarY[t][a][e] += eta;
				}

			for(int n = 0; n < Znodes; ++n)
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

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			{
			double eta = etaYE[t][e]; 

			for(int a = 0; a < Zags; ++a)
				{
				YT[t][a][e] += eta * YEo[t][e][a];
				ZVarY[t][a][e] += eta;
				}
			}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int e = 0; e < Zedges; ++e)
				YT[t][a][e] /= ZVarY[t][a][e];

	for(int t = 0; t < Ztime+1; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				XT[t][a][n] /= ZVarX[t][a][n];

	// Reducing the total cost

	double current_cost = 0.;
	int feasible_edge_count = 0;

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int e = 0; e < Zedges; ++e)
				{
				auto [n1,n2] = edges[e];

				if (n1 == n2 && n1 == Tasks[a].goal)
					continue;

				if (FX[t][a][n1] == 0 || FX[t+1][a][n2] == 0)
					continue;

				current_cost += YT[t][a][e];
				feasible_edge_count++;				
				}

	double Total_Cost = 354.;

	if (current_cost > Total_Cost)
		for(int t = 0; t < Ztime; ++t)
			for(int a = 0; a < Zags; ++a)
				for(int e = 0; e < Zedges; ++e)
					{
					auto [n1,n2] = edges[e];

					if (n1 == n2 && n1 == Tasks[a].goal)
						continue;

					if (FX[t][a][n1] == 0 || FX[t+1][a][n2] == 0)
						{
						YT[t][a][e] = 0.;
						continue;
						}

					YT[t][a][e] -= (current_cost - Total_Cost)/feasible_edge_count;
					}

	changeVar(XCB, YCB, XGB, YEB);
	}

void RRR()
	{    
	int totVar, totCons;
	double diff, avgerr;

	projA(XC, YC, XG, YE);
	reflect(XCA, YCA, XGA, YEA);
	projB(XCR, YCR, XGR, YER);

	tXGerr = 0.;
	tYEerr = 0.;
	tXYerr = 0.;
	toterr = 0.;
	avgerr = 0.;
	TXerr = 0.;
	TYerr = 0.;

	totVar = 0; 
	totCons = 0; 


	// Consistency Constraint between X and Y variables 

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			{
			errXY[t][a] = 0.;

			for(int e = 0; e < Zedges; ++e)
				{
				diff = YCB[t][a][e] - YCA[t][a][e];
				YC[t][a][e] += beta*diff;
				errXY[t][a] += sq(diff);
				}

			TYerr += errXY[t][a]; //////////////////////////////////////////////

			for(int n = 0; n < Znodes; ++n)
				{
				diff = XCB[t][a][n][0] - XCA[t][a][n][0];
				XC[t][a][n][0] += beta*diff;
				errXY[t][a] += sq(diff);
				TXerr += sq(diff); //////////////////////////////////////////////

				diff = XCB[t][a][n][1] - XCA[t][a][n][1];
				XC[t][a][n][1] += beta*diff;
				errXY[t][a] += sq(diff);

				TXerr += sq(diff); //////////////////////////////////////////////
				}

			tXYerr += errXY[t][a];
			errXY[t][a] /= (Zedges + 2. * Znodes);
			avgerr += errXY[t][a];
			}

        totVar += Ztime * Zags * (Zedges + 2. * Znodes);
        toterr += tXYerr;

        tXYerr /= Ztime * Zags * (Zedges + 2. * Znodes);
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
	TXerr += tXGerr;

        tXGerr /= (Ztime+1)*Zags*Znodes;
        totCons += (Ztime+1)*Znodes;

	// No crossing conflict resolution 

	for(int t = 0; t < Ztime; ++t)
		{
		int e = 0;
		while (e < Zedges)
			{
			auto [n1,n2] = edges[e];

			if (n1 < n2)
				{
				assert(e + 1 < Zedges);
				// combined agent vector for the (n1,n2) and (n2,n1) edges should be either zero or 1-hot vector 
				totCons++;
				
				errYE[t][e] = 0.;
				errYE[t][e+1] = 0.;

				for(int a = 0; a < Zags; ++a)
					{
					diff = YEB[t][e][a] - YEA[t][e][a];
					YE[t][e][a] += beta*diff;
					errYE[t][e] += sq(diff);
					}

				for(int a = 0; a < Zags; ++a)
					{
					diff = YEB[t][e+1][a] - YEA[t][e+1][a];
					YE[t][e+1][a] += beta*diff;
					errYE[t][e] += sq(diff);
					}

				tYEerr += errYE[t][e];
				errYE[t][e] /= Zags*2.;
				avgerr += errYE[t][e];

				//errYE[t][e+1] = errYE[t][e];

				e += 2;
				}

			else if (n1 == n2)
				{ 
				totCons++;

				errYE[t][e] = 0.;

				for(int a = 0; a < Zags; ++a)
					{
					diff = YEB[t][e][a] - YEA[t][e][a];
					YE[t][e][a] += beta*diff;
					errYE[t][e] += sq(diff);
					}

				tYEerr += errYE[t][e];
				errYE[t][e] /= Zags;
				avgerr += errYE[t][e];

				e++;
				}

			else 
				e++;

			}
		}

        totVar += Ztime*Zedges*Zags;
        toterr += tYEerr;

	TYerr += tYEerr;
        tYEerr /= Ztime*Zedges*Zags;

	//totCons already added in the loop

	toterr /= totVar;
	avgerr /= totCons;

	// Tuning Variable weights instead of constraints 

	TXerr /= (3.*Ztime + 1.)*Zags*Znodes;
	TYerr /= 2.*Ztime*Zags*Zedges;

	//etaX *= (1. + epsilon*((2.*TXerr/(TXerr+TYerr)) - 1.));
	//etaY *= (1. + epsilon*((2.*TYerr/(TXerr+TYerr)) - 1.));

	etaX += epsilon*((2.*TXerr/(TXerr+TYerr)) - etaX);
	etaY += epsilon*((2.*TYerr/(TXerr+TYerr)) - etaY);

	TXerr = sqrt(TXerr);
	TYerr = sqrt(TYerr);
/*
        // weight tuning 
	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			etaXY[t][a] += epsilon*((errXY[t][a]/avgerr) - etaXY[t][a]);
			//etaXY[t][a] *= (1. + epsilon*((errXY[t][a]/avgerr) - 1.));

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			etaXG[t][n] += epsilon*((errXG[t][n]/avgerr) - etaXG[t][n]);
			//etaXG[t][n] *= (1. + epsilon*((errXG[t][n]/avgerr) - 1.));

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			{
			auto [n1,n2] = edges[e];
			if (n1 <= n2)
				etaYE[t][e] += epsilon*((errYE[t][e]/avgerr) - etaYE[t][e]);
				//etaYE[t][e] *= (1. + epsilon*((errYE[t][e]/avgerr) - 1.));
			}
*/

	tXGerr = sqrt(tXGerr);
	tYEerr = sqrt(tYEerr);
	tXYerr = sqrt(tXYerr);

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
				 << tXYerr << ' '
				 << tXGerr << ' '
				 << tYEerr << ' '
				 << '\t'<< TXerr << ' '
				 << '\t'<< TYerr << ' '
				 << '\t'<< etaX << ' '
				 << '\t'<< etaY << ' '
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
			etaXY[t][a] = 1.;

	for(int t = 0; t < Ztime+1; ++t)
		for(int n = 0; n < Znodes; ++n)
			etaXG[t][n] = 1.;

	for(int t = 0; t < Ztime; ++t)
		for(int e = 0; e < Zedges; ++e)
			etaYE[t][e] = 1.;

	etaX = 1.;
	etaY = 1.;
	}

void init()
	{

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int e = 0; e < Zedges; ++e)
				YT[t][a][e] = urand(-1.0,1.0);

	for(int t = 1; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				XT[t][a][n] = urand(-1.0,1.0);

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

	changeVar(XC, YC, XG, YE);

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
			std::fill_n(YT[t][a], Zedges, 0.0);

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
			t++;
			}

		++agent_id; // next agent line
		}


	for (int t = 0; t < Ztime; ++t) 
		for (int a = 0; a < Zags; ++a) 
			for(int e = 0; e < Zedges; ++e)
				{
				auto [n1,n2] = edges[e];
				if (XT[t][a][n1] > 0.5 && XT[t+1][a][n2] > 0.5)
					YT[t][a][e] = 1.0;
				}

	for(int t = 0; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int e = 0; e < Zedges; ++e)
				YT[t][a][e] += lambda * urand(-1.0,1.0);

	for(int t = 1; t < Ztime; ++t)
		for(int a = 0; a < Zags; ++a)
			for(int n = 0; n < Znodes; ++n)
				XT[t][a][n] += lambda * urand(-1.0,1.0);

	changeVar(XC, YC, XG, YE);

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
		init();
		//initSol("solution.txt", 0.0);

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
			printf("%d \t %d \n",i,iter_needed);
			}
		else       // failure path: write -1 as in C code
			{
			double iterpersec = maxiter / deltat;
			stat << -1  << '\t' << deltat << '\t' << iterpersec << '\n';
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
	if (argc != 13) 
		{
		std::cerr
		<< "Usage:\n  " << argv[0]
		<< " <mapPath> <scenarioPath> <Zags> <runID> <total_time> <beta> <maxiter> <iterstride> <stoperr> <epsilon> <seed> <numrun>\n";
		return EXIT_FAILURE;
		}

	// Parse args 
	const std::string mapPath  	= argv[1];
	const std::string scenarioPath 	= argv[2];
	Zags	 			= std::stoi(argv[3]);
	const std::string runID        	= argv[4];
	int total_time			= std::stoi(argv[5]);
	beta  			    	= std::stod(argv[6]);
	const int maxiter		= std::stoi(argv[7]);
	const int iterstride		= std::stoi(argv[8]);
	const double stoperr   		= std::stod(argv[9]);
	epsilon 			= std::stod(argv[10]);
	const int seed		      	= std::stoi(argv[11]);
	const int numrun		= std::stoi(argv[12]);

	Ztime = total_time;

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
