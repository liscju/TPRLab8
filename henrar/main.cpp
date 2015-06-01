#include <iostream>

#DEFINE BSZ (16)

void update (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
	
	if (I>=N*N)
	{
		return;
	}	

	float u_prev_sh[BSZ][BSZ];

	u_prev_sh[i][j] = u_prev[I];
	
	
	bool bound_check = ((I>N) && (I< N*N-1-N) && (I%N!=0) && (I%N!=N-1)); 
	bool block_check = ((i>0) && (i<BSZ-1) && (j>0) && (j<BSZ-1));
 
	if (block_check)
	{	
		u[I] = u_prev_sh[i][j] + alpha*dt/h/h * (u_prev_sh[i+1][j] + u_prev_sh[i-1][j] + u_prev_sh[i][j+1] + u_prev_sh[i][j-1] - 4*u_prev_sh[i][j]);
	}
	else if (bound_check) 
	{	
		u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
	} 								}
} 

int main()
{
	int N = 128;
	int blockSize = BSZ;
	
	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;
	int steps = (int) ceil(time/dt);
	int I;
	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];

	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = 200.0f;}
		}
	}
	
	
	return 0;
}
