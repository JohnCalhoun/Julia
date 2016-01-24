CC=nvcc
OPTIONS=-I /usr/local/include --std=c++11 

julia:julia.cu functions.cu functions.h
	$(CC) -o julia julia.cu $(OPTIONS) --optimize 3 --use_fast_math

