#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <chrono>

using namespace std;

static long long cntSteps = 1E6;
double a = 0.001, b = 0.01;
double step = (b - a)/(double)cntSteps;
int N_THREADS;

double f(double x){
	return (1/(x * x)) * sin(1/x) * sin(1/x); 
}

void sequential_solution(){
	double x = 0, ans, sum = 0.0;
	for (int i = 0; i < cntSteps; ++i){
		x = a + (i + 0.5) * step;
		sum += f(x); 
	}
	ans = sum * step;
}

void atomic_solution(){
	double x = 0, ans, sum = 0.0;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for
	for (int i = 0; i < cntSteps; ++i){
		x = a + (i + 0.5) * step;
		double cur_ans = f(x);
		#pragma omp atomic
		sum += cur_ans; 
	}
	ans = sum * step;
}

void critical_solution(){
	double x = 0, ans, sum = 0.0;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel for
	for (int i = 0; i < cntSteps; ++i){
		x = a + (i + 0.5) * step;
		double cur_ans = f(x);
		#pragma omp critical
		sum += cur_ans; 
	}
	ans = sum * step;
}

void lock_solution(){
	double x = 0, ans, sum = 0.0;
	omp_set_num_threads(N_THREADS);
	omp_lock_t lock;
	omp_init_lock(&lock);
	#pragma omp parallel for
	for (int i = 0; i < cntSteps; ++i){
		omp_set_lock (&lock);
		x = a + (i + 0.5) * step;
		sum += f(x); 
		omp_unset_lock(&lock);
	}
	ans = sum * step;
	omp_destroy_lock(&lock);
}

void reduction_solution(){
	double x = 0, ans, sum = 0.0;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel
	{
		double x;
		#pragma omp for reduction(+:sum)
		for (int i = 0; i < cntSteps; ++i){
			x = a + (i + 0.5) * step;
			sum += f(x); 
		}
		ans = sum * step;
	}
}

int main(int argc, char** argv){
	freopen("output.txt", "w", stdout);
	int threads_arr [4]  {1, 2, 4, 8};
	for (int i = 0; i < 4; i++){
		N_THREADS = threads_arr[i];
		auto start = chrono::high_resolution_clock::now(), end = chrono::high_resolution_clock::now();
		
		// Sequential solution
		start = chrono::high_resolution_clock::now();
		sequential_solution();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_seq = end - start;
		
		// Solution with help of atomic
		start = chrono::high_resolution_clock::now();
		atomic_solution();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_atomic = end - start;
		
		// Solution with help of ciritical section
		start = chrono::high_resolution_clock::now();
		critical_solution();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_critical = end - start;
		
		// Solution with help of locks
		start = chrono::high_resolution_clock::now();
		lock_solution();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_lock = end - start;
		
		// Solution with help of reduction
		start = chrono::high_resolution_clock::now();
		reduction_solution();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_reduction = end - start;
		
		printf("%d threads\n", N_THREADS);
		printf("%.5lf\n", ans_seq.count());
		printf("%.5lf\n", ans_atomic.count());
		printf("%.5lf\n", ans_critical.count());
		printf("%.5lf\n", ans_lock.count());
		printf("%.5lf\n\n", ans_reduction.count());
	}
	return 0;
}
