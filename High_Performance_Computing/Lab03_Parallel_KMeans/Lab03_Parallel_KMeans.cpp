#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <math.h>
#include <time.h>
#include <limits>

using namespace std;

const int NMAX = 5000, N_CLUSTERS = 15;
int n = 5000;
struct point{
	double x = 0, y = 0;
};
point a[NMAX], centers[N_CLUSTERS];
int cluster[NMAX], new_cluster[NMAX];
double cur_distance[N_CLUSTERS], new_distance[NMAX];
double max_cord = 0, min_cord = LLONG_MAX;

void init_centers(){
	for (int i = 0; i < N_CLUSTERS; ++i){
		centers[i] = a[i];
	}
}

void init_clusters(){
	for (int i = 0; i < n; ++i){
		cluster[i] = rand()%N_CLUSTERS;
		new_cluster[i] = rand()%N_CLUSTERS;
	}
}

double distance(point p1, point p2){
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

void update_clusters(){
	for (int i = 0; i < n; ++i){
		cluster[i] = new_cluster[i];
		new_cluster[i] = 0;
	}
}

bool condition_to_convergence(){
	for (int i = 0; i < N_CLUSTERS; ++i){
		if (cur_distance[i] != new_distance[i]){
			return false;
		}
	}
	return true;
}

void update_cluster_centers(){
	for (int i = 0; i < N_CLUSTERS; ++i){
		point new_center;
		int count = 0;
		for (int j = 0; j < n; ++j){
			if (cluster[j] == i){
				count ++;
				new_center.x += a[j].x;
				new_center.y += a[j].y;
			}
		}
		new_center.x /= count;
		new_center.y /= count;
		centers[i] = new_center;
	}
}
	
void update_inner_distance(){
	for (int i = 0; i < n; ++i){
		new_distance[cluster[i]] += distance(centers[cluster[i]], a[i]);
	}
}

void update_distance(){
	for (int i = 0; i < n; ++i){
		cur_distance[i] = new_distance[i];
		new_distance[i] = 0;
	}
}

void print_arr(){	
	for (int i = 0; i < n; ++i){
		cout << cluster[i] << ' ' << new_cluster[i] <<  endl;
	}
}

void sequential_kmeans(){	
	init_centers();
	init_clusters();
	int iter = 0;
	while (true){
		iter ++;
		for (int i = 0; i < n; ++i){
			int clust_num = 0;
			double dist = numeric_limits<double>::max();
			for (int j = 0; j < N_CLUSTERS; j++){
				double cur_dist = distance(a[i], centers[j]);
				if (cur_dist < dist){
					dist = cur_dist;
					clust_num = j;
				}
			}
			cluster[i] = clust_num;
		}
		update_cluster_centers();
		update_inner_distance();
		if (condition_to_convergence()){
			break;
		}
		update_distance();
	}	
}

void parallel_kmeans(){
	init_centers();
	init_clusters();
	int iter = 0;
	while (true){
		iter ++;
		#pragma omp parallel for 
		for (int i = 0; i < n; ++i){
			int clust_num = 0;
			double dist = numeric_limits<double>::max();
			for (int j = 0; j < N_CLUSTERS; j++){
				double cur_dist = distance(a[i], centers[j]);
				if (cur_dist < dist){
					dist = cur_dist;
					clust_num = j;
				}
			}
			cluster[i] = clust_num;
		}
		update_cluster_centers();
		update_inner_distance();
		if (condition_to_convergence()){
			break;
		}
		update_distance();
	}
}

int main(int argc, char** argv){
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	srand(time(NULL));
	for (int i = 0; i < n; ++i){
		cin >> a[i].x >> a[i].y;
	}
	
	auto start = chrono::high_resolution_clock::now(), end = chrono::high_resolution_clock::now();
	start = chrono::high_resolution_clock::now();
	sequential_kmeans();
	end = chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ans_seq = end - start;
	printf("%.5lf\n", ans_seq.count());

	for (int i = 1; i <= 4; ++i){
		omp_set_num_threads(i);
		start = chrono::high_resolution_clock::now();
		parallel_kmeans();
		end = chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ans_parallel = end - start;
		printf("%.5lf\n", ans_parallel.count());
	}
	
	return 0;
}
