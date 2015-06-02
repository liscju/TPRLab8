#!/bin/bash

# TODO: ustalić zbiór parametrów (!!! N powinno być wielokrotnością 32 !!!)
nSizes=(10 100)

for nSize in "${nSizes[@]}"
do
	#TODO: ustalić wywołania dla pozostałych wersji GPU
	./cpu.out $nSize >> output/out_cpu.txt
done

