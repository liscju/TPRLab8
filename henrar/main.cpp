#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
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

void updateLaplace (float *u, float *u_prev, int N, float h, float dt, float alpha) {
	for(int i=1; i<N; ++i) {
		for(int j=1; j<N; ++j) {
			int I = j*N + i;

			if (I >= N*N) {
				continue;
			}	

			u_prev[I] = u[I];

			if ( (I > N) && (I < N*N - 1 - N) && (I % N != 0) && (I % N != N - 1)) {	
				u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
			}			
		}
	}
}

double get_time() {  
	struct timeval tim;
	gettimeofday(&tim, NULL);
	return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

int main(int argc, char** argv) {

	std::ofstream resultsFile("results/results_cpu.txt", std::ofstream::app);
	int N;
	for (int hurr = 0; hurr < 8; hurr++)
	{
		N = N_table[hurr];

		float xmin = 0.0f;
		float xmax = 3.5f;
		float ymin = 0.0f;
		float ymax = 2.0f;
		float h = (xmax - xmin) / (N - 1);
		float dt = 0.00001f;
		float alpha = 0.645f;
		float time = 0.4f;
		int steps = (int)ceil(time / dt);
		int I;
		float *x = new float[N*N];
		float *y = new float[N*N];
		float *u = new float[N*N];
		float *u_prev = new float[N*N];

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
		double start = get_time();
		for (int t = 0; t < steps; t++) {
			updateLaplace(u, u_prev, N, h, dt, alpha);
		}

		double stop = get_time();
		double elapsed = stop - start;

		resultsFile << N << "\t" << elapsed << std::endl;

		delete[] x;
		delete[] y;
		delete[] u;
		delete[] u_prev;
	}
	resultsFile.close();
	return EXIT_SUCCESS;
}
