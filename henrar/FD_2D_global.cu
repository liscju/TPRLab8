/*** Calculating a derivative with CD ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <winsock2.h>
#include <cstdint>

const int N_table[10] = { 32, 64, 128, 256, 512, 1024, 2048, 4096 };

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

void checkErrors(char *label) {
	// we need to synchronise first to catch errors due to
	// asynchroneous operations that would otherwise
	// potentially go unnoticed
	cudaError_t err;
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		char *e = (char*) cudaGetErrorString(err);
		fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		char *e = (char*) cudaGetErrorString(err);
		fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
	}
}
	
double get_time() {  
	struct timeval tim;
	cudaThreadSynchronize();
	gettimeofday(&tim, NULL);
	return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

__global__ void copy_array(float *u, float *u_prev, int N, int BSZ) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	if (I>=N*N){
		return;
	}	
	u_prev[I] = u[I];
}

// GPU kernel
__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ) {
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	
	if (I>=N*N){
		return;
	}	
	//if (()>=N || j>){return;}	

	
	// if not boundary do
	if ( (I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1)) {	
		u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
	}
	
	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}

int main(int argc, char** argv) 
{
	std::ofstream resultsFile("results/results_global.txt", std::ofstream::app);

	for (int hurr = 0; hurr < 8; hurr++)
	{
		int N = N_table[hurr];
		int BLOCKSIZE = 16;

		cudaSetDevice(2);

		float xmin = 0.0f;
		float xmax = 3.5f;
		float ymin = 0.0f;
		//float ymax 	= 2.0f;
		float h = (xmax - xmin) / (N - 1);
		float dt = 0.00001f;
		float alpha = 0.645f;
		float time = 0.4f;

		int steps = ceil(time / dt);
		int I;

		float *x = new float[N*N];
		float *y = new float[N*N];
		float *u = new float[N*N];
		float *u_prev = new float[N*N];

		// Generate mesh and intial condition
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				I = N*j + i;
				x[I] = xmin + h*i;
				y[I] = ymin + h*j;
				u[I] = 0.0f;
				if ((i == 0) || (j == 0)) {
					u[I] = 200.0f;
				}
			}
		}

		// Allocate in GPU
		float *u_d, *u_prev_d;

		cudaMalloc((void**)&u_d, N*N*sizeof(float));
		cudaMalloc((void**)&u_prev_d, N*N*sizeof(float));

		// Copy to GPU
		cudaMemcpy(u_d, u, N*N*sizeof(float), cudaMemcpyHostToDevice);

		// Loop 
		dim3 dimGrid(int((N - 0.5) / BLOCKSIZE) + 1, int((N - 0.5) / BLOCKSIZE) + 1);
		dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
		double start = get_time();
		for (int t = 0; t < steps; t++) {
			copy_array << <dimGrid, dimBlock >> > (u_d, u_prev_d, N, BLOCKSIZE);
			update << <dimGrid, dimBlock >> > (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
		}
		double stop = get_time();
		checkErrors("update");

		double elapsed = stop - start;
		//	!!! Wydruk na konsolę: N time
		resultsFile << N << "\t" << elapsed << std::endl;

		// Copy result back to host
		cudaMemcpy(u, u_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

		// Free device
		delete[] x;
		delete[] y;
		delete[] u;
		delete[] u_prev;
		cudaFree(u_d);
		cudaFree(u_prev_d);
	}
	resultsFile.close();
	return 0;
}
