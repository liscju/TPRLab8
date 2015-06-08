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

void checkErrors(char *label) 
{
	cudaError_t err;
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) 
	{
		char *e = (char*) cudaGetErrorString(err);
		fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		char *e = (char*) cudaGetErrorString(err);
		fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
	}
}

double get_time() 
{ 
	struct timeval tim;
	cudaThreadSynchronize();
	gettimeofday(&tim, NULL);
	return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}


texture<float, 2> tex_u;
texture<float, 2> tex_u_prev;

// GPU kernels
__global__ void copy_kernel (float *u, float *u_prev, int N, int BSZ, int N_max) 
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BSZ;
	int y = j + blockIdx.y*BSZ;
	int I = x + y*N_max;
		
	float value = tex2D(tex_u, x, y);

	u_prev[I] = value;
}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ, int N_max) 
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x*BSZ;
	int y = j + blockIdx.y*BSZ;
	int I = x + y*N_max;

	float t, b, r, l, c;
	c = tex2D(tex_u_prev, x, y);	
	t = tex2D(tex_u_prev, x, y+1);	
	b = tex2D(tex_u_prev, x, y-1);	
	r = tex2D(tex_u_prev, x+1, y);	
	l = tex2D(tex_u_prev, x-1, y);

	if ( (x!=0) && (y!=0) && (x!=N-1) && (y!=N-1)) 
	{
		u[I] = c + alpha*dt/h/h * (t + b + l + r - 4*c);	
	}
}

int main(int argc, char** argv) 
{
	std::ofstream resultsFile("results/results_texture.txt", std::ofstream::app);

	int N;
	int BLOCKSIZE = 32;
	float xmin = 0.0f;
	float xmax = 3.5f;
	float ymin = 0.0f;
	float dt = 0.00001f;
	float alpha = 0.645f;
	float time = 0.4f;

	for (int hurr = 0; hurr < 8; hurr++)
	{
		N = N_table[hurr];	// For textures to work, N needs to be a multiple of
			
		int N_max = (int((N - 0.5) / BLOCKSIZE) + 1) * BLOCKSIZE;
		float h = (xmax - xmin) / (N - 1);
		int steps = (int)ceil(time / dt);
		int I;

		float *x = new float[N*N];
		float *y = new float[N*N];
		float *u = new float[N_max*N_max];
		float *u_prev = new float[N*N];

		for (int j = 0; j < N_max; j++) 
		{
			for (int i = 0; i < N_max; i++) 
			{
				I = N_max*j + i;
				u[I] = 0.0f;
				if (((i == 0) || (j == 0)) && (j < N) && (i < N)) 
				{
					u[I] = 200.0f;
				}
			}
		}

		for (int j = 0; j < N; j++) 
		{
			for (int i = 0; i < N; i++) 
			{
				I = N*j + i;
				x[I] = xmin + h*i;
				y[I] = ymin + h*j;
			}
		}

		float *u_d, *u_prev_d;

		cudaMalloc((void**)&u_d, N_max*N_max*sizeof(float));
		cudaMalloc((void**)&u_prev_d, N_max*N_max*sizeof(float));

		// Bind textures
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(NULL, tex_u, u_d, desc, N_max, N_max, sizeof(float)*N_max);
		cudaBindTexture2D(NULL, tex_u_prev, u_prev_d, desc, N_max, N_max, sizeof(float)*N_max);

		// Copy to GPU
		cudaMemcpy(u_d, u, N_max*N_max*sizeof(float), cudaMemcpyHostToDevice);

		// Loop 
		dim3 dimGrid(int((N_max - 0.5) / BLOCKSIZE) + 1, int((N_max - 0.5) / BLOCKSIZE) + 1);
		dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
		double start = get_time();
		for (int t = 0; t < steps; t++) 
		{
			copy_kernel << <dimGrid, dimBlock >> > (u_d, u_prev_d, N, BLOCKSIZE, N_max);
			update << <dimGrid, dimBlock >> > (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE, N_max);

		}
		double stop = get_time();
		checkErrors("update");

		double elapsed = stop - start;

		resultsFile << N << "\t" << elapsed << std::endl;

		cudaMemcpy(u, u_d, N_max*N_max*sizeof(float), cudaMemcpyDeviceToHost);

		delete[] x;
		delete[] y;
		delete[] u;
		delete[] u_prev;
		cudaUnbindTexture(tex_u);
		cudaUnbindTexture(tex_u_prev);
		cudaFree(u_d);
		cudaFree(u_prev_d);
	}
	resultsFile.close();
	return 0;
}
