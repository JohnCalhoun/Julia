#include <stdio.h>
#include <cuda.h>
#include <qdbmp/qdbmp.h>
#include "functions.h"
#include <string>
#define __both__ __device__ __host__

#define DIM 10000
#define ESCAPE 1000
#define WIDTH 1
#define HEIGHT 1
#define Creal .8
#define Cimag .156
#define ITERATIONS 40

int main() {	
	BMP*			bmp;
	const USHORT	depth=8;
	
	typedef unsigned int		Cor;
	typedef int				Result;
	typedef double				Value;
	typedef unsigned long		Size;
	
	const Size totalPixels	=Size(DIM)*Size(DIM);
	
	const Size blockSize	=1000000;
	const int numberOfBlocks =totalPixels/blockSize;

	const Size cordinate_size	=blockSize*sizeof(Cor);
	const Size value_size		=blockSize*sizeof(Value);
	const Size result_size		=blockSize*sizeof(Result);

	Cor*		cordinate_x_d;
	Cor*		cordinate_y_d;
	Value*	real_d;
	Value*	imag_d;
	Result*	result_d;
	
	Cor*		cordinate_x_h;
	Cor*		cordinate_y_h;
	Result*	result_h;
	
	#define DEVICEMALLOC cudaMalloc
	DEVICEMALLOC( (void**)&cordinate_x_d,cordinate_size);
	DEVICEMALLOC( (void**)&cordinate_y_d,cordinate_size);
	DEVICEMALLOC( (void**)&real_d,value_size);
	DEVICEMALLOC( (void**)&imag_d,value_size);
	DEVICEMALLOC( (void**)&result_d,result_size);
	#undef DEVICEMALLOC
	
	cordinate_x_h	=(Cor*)malloc(cordinate_size); 
	cordinate_y_h	=(Cor*)malloc(cordinate_size);
	result_h		=(Result*)malloc(result_size);
	
	bmp=BMP_Create(DIM,DIM,depth);
	const Julia::Complex<Value> constant(Creal,Cimag);
	
	for(int i=0; i<numberOfBlocks; i++){
		Julia::blockRender(	
					bmp,
					
					cordinate_x_d,
					cordinate_y_d,
					real_d,
					imag_d,
					result_d,
					
					cordinate_x_h,
					cordinate_y_h,
					result_h,

					Size(i*blockSize),
					blockSize,
					constant,
					ESCAPE,
					ITERATIONS,
					DIM,
					WIDTH,
					HEIGHT
					);
	}

	BMP_WriteFile(bmp,"julia.bmp");
	
	BMP_Free(bmp);
	cudaFree(cordinate_x_d);
	cudaFree(cordinate_y_d);
	cudaFree(real_d);
	cudaFree(imag_d);
	cudaFree(result_d);
	
	free(cordinate_x_h);
	free(cordinate_y_h);
	free(result_h);

	BMP_CHECK_ERROR(stderr,-2);
		return 0;	
}






