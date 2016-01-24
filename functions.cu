#define __both__ __device__ __host__
#include<cuda_occupancy.h>
#include<thrust/for_each.h>
#include<thrust/execution_policy.h>
#include<thrust/iterator/counting_iterator.h>

template<typename T>
__both__ Complex<T> Complex<T>::operator+(const Complex<T>& other)const{
	T r=real+other.Real();
	T i=imag+other.Imag();
	Complex tmp(r,i);
	return tmp; 
};
template<typename T>
__both__ Complex<T> Complex<T>::operator*(const Complex<T>& other)const{
	T r=real*other.Real()-imag*other.Imag();
	T i=real*other.Imag()+imag*other.Real(); 
	Complex tmp(r,i);
	return tmp; 
};	
template<typename T>
__both__ T Complex<T>::abs()const{
	T tmp=sqrt(real*real+imag*imag);
	return tmp;
};


template<typename T,typename S, typename L>
__device__ bool julia(T& r,T& i,const Complex<L>& c,const S escape,const S iterations ){
	int count=0;
	typedef Complex<L> complex; 

	complex point(r,i); 
	bool result=point.abs()<escape;

	while(result and count<iterations){
		point=point*point+c;
		count++;
		result=point.abs()<escape;
	}
	r=point.Real();
	i=point.Imag();
	return result;
}

template<typename T,typename L,typename R,typename S,typename C>
__global__ void setPixels_init(	T* const x_cor,
							T* const y_cor,
							L* const r_value,
							L* const i_value,
							R* const results,
							const S offset,
							const S size,
							const Complex<L> constant,
							const C escape, 
							const C iterations,
							const C dimension,
							const C width,
							const C height){
	typedef T INT;
	typedef L Value;

	INT tid=blockIdx.x*blockDim.x+threadIdx.x;
	
	const Value scalex	=Value(width)	/Value(dimension);
	const Value offsetx	=Value(height)	/Value(2);
	const Value scaley	=Value(width)	/Value(dimension);
	const Value offsety	=Value(height)	/Value(2);

	while(tid<(size) ){
		INT y=(tid+offset)/dimension;
		INT x=(tid+offset)%dimension;

		Value r=x*scalex-offsetx;
		Value i=y*scaley-offsety;
		int result=julia(r,i,constant,escape,iterations);	

		x_cor[tid]=x;
		y_cor[tid]=y;
		r_value[tid]=r;
		i_value[tid]=i;
		results[tid]=result;
		tid+=blockDim.x*gridDim.x;
	}
};

template<typename T,typename R,typename S,typename C>
__global__ void setPixels_step(	T* const r_value,
							T* const i_value,
							R* const results,
							const S count,
							const Complex<T> constant,
							const C escape,
							const C iterations){
	typedef unsigned int INT;
	typedef T	Value;

	INT tid=blockIdx.x*blockDim.x+threadIdx.x;
	while(tid<count ){

		Value r=r_value[tid];
		Value i=i_value[tid];

		int result=julia(r,i,constant,escape,iterations);	

		r_value[tid]=r;
		i_value[tid]=i;
		results[tid]=result;
		tid+=blockDim.x*gridDim.x;
	}
};

template<typename T,typename R, typename S>
void pixelsToBMP(	T* const x_cor,
				T* const y_cor,
				R* const results,
				BMP* const bmp, 
				const S count){
	
	thrust::counting_iterator<S> first(0);
	thrust::counting_iterator<S> last=first+count;

	thrust::for_each(	thrust::seq,
					first,
					last,
					[&](S i){
						BMP_SetPixelIndex(	bmp,
										x_cor[i],
										y_cor[i],
										results[i]
									);
					}		
				);
	//true->1
	//false->0
	BMP_SetPaletteColor(bmp,1,255,255,255); //white
	BMP_SetPaletteColor(bmp,0,0,0,0);		//black
};

template<typename T,typename V,typename R,typename S,typename C>
void blockRender(	BMP* const bmp,
				T* const  x_dev,
				T* const	y_dev,
				V* const	r_dev,
				V* const	i_dev,
				R* const	results_dev,

				T* const	x_host,
				T* const	y_host,
				R* const	results_host,

				const S	offset,
				const S	count,
				const Complex<V> constant,
				const C	escape,
				const C	iterations,
				
				const C dimension, 
				const C width,
				const C height
				){
	const int blockSize_init=512;
	const int gridSize_init=512;

	const int blockSize_step=512;
	const int gridSize_step=512;	
	
	setPixels_init<<<gridSize_init,blockSize_init>>>(	
							x_dev,
							y_dev,
							r_dev,
							i_dev,
							results_dev,
							offset,
							count,
							constant,
							escape,
							iterations,
							dimension,
							width,
							height
						);
	S newSize=count;
	S oldSize=newSize;
	
	S cordinate_size;	
	S result_size; 

	for(int i=0; i<9; i++){
		//calculate new size
		newSize=thrust::count(	thrust::device,
							results_dev,
							results_dev+oldSize,
							1
						);
		cordinate_size=newSize*sizeof(T);	
		result_size=newSize*sizeof(R); 
		if(newSize==0)
			break;
		//remove
		thrust::logical_not<R> op;
		thrust::remove_if(	thrust::device,
						r_dev,
						r_dev+oldSize,
						results_dev,
						op
						);
		thrust::remove_if(	thrust::device,
						i_dev,
						i_dev+oldSize,
						results_dev,
						op
						);
		thrust::remove_if(	thrust::device,
						x_dev,
						x_dev+oldSize,
						results_dev,
						op
						);
		thrust::remove_if(	thrust::device,
						y_dev,
						y_dev+oldSize,
						results_dev,
						op
						);

		thrust::remove(	thrust::device,
						results_dev,
						results_dev+oldSize,
						0
						);
		oldSize=newSize;
		setPixels_step<<<gridSize_step,blockSize_step>>>(	
								r_dev,
								i_dev,
								results_dev,
								newSize,
								constant,
								escape,
								iterations); 
	}
	
	cudaDeviceSynchronize();

	cudaMemcpy(x_host,x_dev,cordinate_size,cudaMemcpyDeviceToHost); 
	cudaMemcpy(y_host,y_dev,cordinate_size,cudaMemcpyDeviceToHost); 
	cudaMemcpy(results_host,results_dev,result_size,cudaMemcpyDeviceToHost);
	
	pixelsToBMP(x_host,y_host,results_host,bmp,newSize);
};








