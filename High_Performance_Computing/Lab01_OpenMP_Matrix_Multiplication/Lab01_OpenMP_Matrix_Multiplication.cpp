#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <pthread.h>
#include <chrono>
#include <random>
#include <thread>

using namespace std;

/*
Size of arrays:
[[400, 400], [700, 700], [1000, 1000]]
Size of threads:
[1, 2, 4, 8, 16]
*/

const int NMAX = 400, MMAX = 400;
int N_THREADS = 0;
int a[MMAX][NMAX], b[NMAX][MMAX], c[MMAX][MMAX];

void init(){
	for (int i = 0; i < MMAX; i++){
		for (int j = 0; j < NMAX; j++){
			a[i][j] = rand()%10; 
		}
	}
	for (int i = 0; i < NMAX; i++){
		for (int j = 0; j < MMAX; j++){
			b[i][j] = rand()%10; 
		}
	}
	for (int i = 0; i < MMAX; i++){
		for (int j = 0; j < MMAX; j++){
			c[i][j] = 0; 
		}
	}
}

void multiply_seq(){
	for (int i = 0; i < MMAX; i++){
		for (int j = 0; j < MMAX; j++){
			for (int k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void multiply_openmp_static(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(static)
	for (i = 0; i < MMAX; i++){
		for (j = 0; j < MMAX; j++){
			for (k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void multiply_openmp_dynamic(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(dynamic)
	for (i = 0; i < MMAX; i++){
		for (j = 0; j < MMAX; j++){
			for (k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void multiply_openmp_guided(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(guided)
	for (i = 0; i < MMAX; i++){
		for (j = 0; j < MMAX; j++){
			for (k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void multiply_openmp_auto(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(auto)
	for (i = 0; i < MMAX; i++){
		for (j = 0; j < MMAX; j++){
			for (k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void multiply_openmp_runtime(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(runtime)
	for (i = 0; i < MMAX; i++){
		for (j = 0; j < MMAX; j++){
			for (k = 0; k < NMAX; k++){
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}


void multiply_openmp_spmd(){
	int i, j, k;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel
	{ 
		int num_thread = omp_get_thread_num();
		for (int i = num_thread * MMAX/N_THREADS; i < min(NMAX, (num_thread + 1) * MMAX/N_THREADS); i++){
			for (int j = 0; j < MMAX; j++){
				for (int k = 0; k < NMAX; k++){
					c[i][j] += a[i][k]*b[k][j];
				}
			}
		}
	}	
}


void print(int arr [NMAX][NMAX]){
	for (int i = 0; i < NMAX; ++i){
		for (int j = 0; j < NMAX; ++j){
			cout << arr[i][j] << ' ';
		}
		cout << endl;
	}
}


int main(){
	freopen("output.txt", "w", stdout);
	
	int threads_arr [5]  {1, 2, 4, 8, 16};
	
	for (int i = 0; i < 5 ; ++i){
		N_THREADS = threads_arr[i];
		auto start = chrono::high_resolution_clock::now(), end = chrono::high_resolution_clock::now();
		// Init all arrays with random numbers
		init();
		// Sequential multiplication	
		start = chrono::high_resolution_clock::now();
		multiply_seq();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_seq = end - start;
		
		// Multiplication with spmd pattern
		start = chrono::high_resolution_clock::now();
		multiply_openmp_spmd();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_spmd = end - start;
		
		// Multiplication with help of openmp_static
		start = chrono::high_resolution_clock::now();
		multiply_openmp_static();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_openmp_static = end - start;
		
		// Multiplication with help of openmp_dynamic
		start = chrono::high_resolution_clock::now();
		multiply_openmp_dynamic();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_openmp_dynamic = end - start;
		
		// Multiplication with help of openmp_guided
		start = chrono::high_resolution_clock::now();
		multiply_openmp_guided();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_openmp_guided = end - start;
		
		// Multiplication with help of openmp_auto
		start = chrono::high_resolution_clock::now();
		multiply_openmp_auto();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_openmp_auto = end - start;
		
		// Multiplication with help of openmp_runtime
		start = chrono::high_resolution_clock::now();
		multiply_openmp_runtime();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_openmp_runtime = end - start;
		
		// Printing results	
		printf("%d threads\n", N_THREADS);
		printf("%.5lf\n", ans_seq.count());
		printf("%.5lf\n", ans_spmd.count());
		printf("%.5lf\n", ans_openmp_static.count());
		printf("%.5lf\n", ans_openmp_dynamic.count());
		printf("%.5lf\n", ans_openmp_guided.count());
		printf("%.5lf\n", ans_openmp_auto.count());
		printf("%.5lf\n\n", ans_openmp_runtime.count());
	}
	return 0;
}
