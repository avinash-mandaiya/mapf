#ifndef ARRAYND_H
#define ARRAYND_H

#include <cstddef>   // for std::size_t

// -------- 3D --------
template<class T>
struct Array3D 
	{
	T ***ptr = nullptr;   // ptr[i][j] -> T*
	T **lvl1 = nullptr;  // bookkeeping for free()
	T *data = nullptr;  // contiguous storage
	int n0=0, n1=0, n2=0;

	void alloc(int a, int b, int c, bool zero=false) 
		{
		if (ptr || lvl1 || data) free_all();   // free old

		n0=a; n1=b; n2=c;
		std::size_t total = std::size_t(a) * std::size_t(b) * std::size_t(c);

		data = zero ? new T[total]{} : new T[total];
		lvl1 = new T*[(std::size_t)a * b];
		ptr  = new T**[a];

		for (int i = 0; i < a; ++i) 
			{
			ptr[i] = lvl1 + (std::size_t)i * b;
			for (int j = 0; j < b; ++j) 
				ptr[i][j] = data + ((std::size_t)i * b + j) * c;
			}
		}

	void fill_row(int t, int n, const T& value) 
		{
		std::fill_n(ptr[t][n], n2, value);
		}

	void free_all() 
		{
		delete[] ptr;  ptr  = nullptr;
		delete[] lvl1; lvl1 = nullptr;
		delete[] data; data = nullptr;
		n0=n1=n2=0;
		}

	T** operator[](int i) { return ptr[i]; }
	const T* const* operator[](int i) const { return ptr[i]; }

	};

// -------- 2D --------
template<class T>
struct Array2D
	{
	T ** ptr  = nullptr;  // ptr[i] -> T*
	T *  data = nullptr;  // contiguous storage
	int n0=0, n1=0;

	void alloc(int a, int b, bool zero=false)
		{
		n0=a; n1=b;
		std::size_t total = (std::size_t)a * b;

		data = zero ? new T[total]{} : new T[total];
		ptr  = new T*[a];

		for (int i = 0; i < a; ++i)
			ptr[i] = data + (std::size_t)i * b;
		}

	void free_all()
		{
		delete[] ptr;  ptr  = nullptr;
		delete[] data; data = nullptr;
		n0=n1=0;
		}

	T* operator[](int i) { return ptr[i]; }
	const T* operator[](int i) const { return ptr[i]; }
	};

#endif // ARRAYND_H

