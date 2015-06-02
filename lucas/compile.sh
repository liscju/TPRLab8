#!/bin/sh
# TODO: sprawdzić czy arch i code są poprawnie ustawione
nvcc FD_2D_global.cu -o gpu_global.out -gencode arch=compute_20,code=sm_20
nvcc FD_2D_shared.cu -o gpu_shared.out -gencode arch=compute_20,code=sm_20
nvcc FD_2D_texture_pad.cu -o gpu_texture_pad.out -gencode arch=compute_20,code=sm_20
g++ main.cpp -o cpu.out
