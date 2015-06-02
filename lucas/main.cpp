#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/time.h>


void updateLaplace (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ) {
	for(int i=1; i<N; ++i) {
		for(int j=1; j<N; ++j) {
			int I = j*N + i;

			if (I >= N*N) {
				std::cout << "DEBUG: I >= N*N" << std::endl;
				continue;
			}	

			u_prev[I] = u[I];

			if ( (I > N) && (I < N*N - 1 - N) && (I % N != 0) && (I % N != N - 1)) {	
				u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
				//std::cout << "DEBUG: (I > N) && (I < N*N - 1 - N) && (I % N != 0) && (I % N != N - 1)" << std::endl;
			}			
		}
	}
}

double get_time() {  
	struct timeval tim;
	gettimeofday(&tim, NULL);
	return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

int main() {
	int N = 128;
	int blockSize = 16;
	
	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	float ymax 	= 2.0f;
	float h   	= (xmax - xmin) / (N - 1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;
	int steps = (int) ceil(time/dt);
	int I;
	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];

	for (int j = 0; j < N; j++) {	
		for (int i=0; i<N; i++) {	
			I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i == 0) || (j == 0)) {
				u[I] = 200.0f;
			}
		}
	}
	double start = get_time();
	std::cout << "DEBUG: Calling update function" << std::endl;
	for (int t=0; t<steps; t++) {
		updateLaplace(u, u_prev, N, h, dt, alpha, blockSize);
	}	
	std::cout << "DEBUG: Exit update function" << std::endl;
	double stop = get_time();
	double elapsed = stop - start;
	std::cout << "time = " << elapsed << std::endl;

	std::ofstream temperature;
	temperature.open("output/cpu_results.txt");
	temperature << "x[I]\ty[I]\tu[I]" << std::endl;
	for (int j = 0; j < N; j++) {	
		for (int i = 0; i < N; i++) {	
			I = N*j + i;
			temperature << x[I] << "\t" <<y [I] << "\t" << u[I] << std::endl;
		}
		temperature << std::endl;
	}

	temperature.close();

	delete[] x;
	delete[] y;
	delete[] u;
	delete[] u_prev;
	
	return 0;
}
