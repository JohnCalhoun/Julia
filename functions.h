#ifndef JULIA_FUNCTIONS_H
#define JULIA_FUNCTIONS_H

#include <stdio.h>
#include <cuda.h>
#include "qdbmp.h"
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define __both__ __device__ __host__

namespace Julia{
	template<typename T>
	class Complex {
		private:
		T real;
		T imag;

		public:
		__both__ Complex(T r,T i):real(r),imag(i){};
		
		__both__ T Real()const{return real;};
		__both__ T Imag()const{return imag;};
		__both__ void Real(T r){real=r;};
		__both__ void Imag(T i){imag=i;};
		
		__both__ Complex operator+(const Complex&)const;
		__both__ Complex operator*(const Complex&)const;
		__both__ T abs()const;
	};

template<typename T,typename L,typename R,typename S,typename C>
	__global__ void setPixels_init(	T* const,
								T* const,
								L* const,
								L* const,
								R* const,
								const S,
								const S,
								const Complex<L>,
								const C,
								const C,
								const C,
								const C,
								const C);

	template<typename T,typename R,typename S,typename C>
	__global__ void setPixels_step(	T* const,
								T* const,
								R* const,
								const S,
								const Complex<T>,
								const C,
								const C);

	template<typename T,typename R,typename S>
	void pixelsToBMP(	T* const, 
					T* const,
					R* const,
					BMP* const,
					const S);

	template<typename T,typename V,typename R,typename S,typename C>
	void blockRender(	BMP* const,
					T* const,
					T* const,
					V* const,
					V* const,
					R* const,

					T* const,
					T* const,
					R* const,

					const S,
					const S,

					const Complex<V>,
					const C,
					const C,

					const C,
					const C,
					const C
					);

	template<typename T,typename S,typename L>
	__device__ bool julia(T&,T&,const Complex<L>&,const S, const S);
	#include"functions.cu"
};//end namespace julia
#endif
