// structs.hpp

#pragma once
#include <cstdint>
#include <ostream>
#include <vector>
#include <algorithm>

struct Pos 
	{
	double x = 0.0;
	double y = 0.0;

	// in-place
	Pos& operator+=(const Pos& o) { x += o.x; y += o.y; return *this; }
	Pos& operator-=(const Pos& o) { x -= o.x; y -= o.y; return *this; }
	Pos& operator*=(double s)     { x *= s;   y *= s;   return *this; }
	Pos& operator/=(double s)     { x /= s;   y /= s;   return *this; }
	};

// non-member (simple, inline)
inline Pos operator+(Pos a, const Pos& b) { a += b; return a; }
inline Pos operator-(Pos a, const Pos& b) { a -= b; return a; }
inline Pos operator*(Pos a, double s)     { a *= s; return a; }
inline Pos operator*(double s, Pos a)     { a *= s; return a; }
inline Pos operator/(Pos a, double s)     { a /= s; return a; }

inline std::ostream& operator<<(std::ostream& os, const Pos& p) 
	{
	return os << "(" << p.x << ", " << p.y << ")";
	}

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

inline void sort_items(std::vector<Item>& a)
	{    
	std::sort(a.begin(), a.end(), [](const Item& x, const Item& y) {return x.val < y.val;}  );   // ascending
	} 
