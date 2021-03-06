#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>
#include <gdf/cffi/functions.h>

#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "bitmaskops.h"

#include <cstring>



//TODO: someone who knows more about cuda what
//do I do to pointers and width so that they are not accessed from global
//not the data inside of the pointers but just their address
struct hash_fnv_array_op : public thrust::unary_function< gdf_size_type, unsigned long long>
{

	void ** pointers;
	int length;
	int * widths;

	hash_fnv_array_op(int _length, void ** _pointers, int * _widths){
		this->length = _length;
		this->pointers = _pointers;
		this->widths = _widths;

	}

	__host__ __device__
	gdf_size_type operator()(gdf_size_type index)
	{

		unsigned long long hash = 14695981039346656037ull;

		//oh god i hope this is reasonable
		for(int i = 0; i < length; i++){
			if(widths[i] == 1){
				char data = *(((char *) pointers[i]) + index);
				for(int j = 0; j < widths[i]; j++){
					hash = hash ^ data;
					hash = hash * 1099511628211;
				}
			}else if(widths[i] == 2){
				short t_data = *(((short *) pointers[i]) + index);
				char * data = (char *) &t_data;
				for(int j = 0; j < widths[i]; j++){
					hash = hash ^ data[j];
					hash = hash * 1099511628211;
				}
			}else if(widths[i] == 4){
				int t_data = *(((int *) pointers[i]) + index);
				char * data = (char *) &t_data;
				for(int j = 0; j < widths[i]; j++){
					hash = hash ^ data[j];
					hash = hash * 1099511628211;
				}
			}else if(widths[i] == 8){
				long long t_data = *(((long long *) pointers[i]) + index);
				char * data = (char *) &t_data;
				for(int j = 0; j < widths[i]; j++){
					hash = hash ^ data[j];
					hash = hash * 1099511628211;
				}
			}
		}
		return hash;
	}
};

gdf_error gpu_hash_columns(gdf_column ** columns_to_hash, int num_columns, gdf_column * output_column, void * stream_pvoid){
	cudaStream_t * stream = (cudaStream_t *)stream_pvoid;
	//TODO: require sizes of columsn to be same and > 0
	//require column output type be int64 even though output is unsigned
	bool created_stream = false;
	cudaStream_t temp_stream;
	if(stream == nullptr){
		stream = &temp_stream;
		cudaStreamCreate(stream);
		created_stream = true;
	}

	gdf_size_type num_values = columns_to_hash[0]->size;

	//copy widths into device memory
	int * widths;
	cudaMalloc(&widths,sizeof(int) * num_columns);
	int * host_widths = new int[num_columns];
	for(int i = 0; i <  num_columns; i++){
		get_column_byte_width(columns_to_hash[i], &host_widths[i]);
	}
	cudaMemcpyAsync(widths,host_widths,sizeof(int) * num_columns,cudaMemcpyHostToDevice,*stream);


	//copy addresses into device memory
	void ** pointers;
	cudaMalloc(&pointers,sizeof(void *) * num_columns);
	void ** data_holder = new void *[num_columns];
	for(int i = 0; i <  num_columns; i++){
		data_holder[i] = columns_to_hash[i]->data;
	}
	cudaMemcpyAsync(pointers,data_holder,sizeof(void *) * num_columns,cudaMemcpyHostToDevice,*stream);


	hash_fnv_array_op op(num_columns,pointers,widths);
	auto begin = thrust::make_counting_iterator<gdf_size_type>(0);
	auto end = thrust::make_counting_iterator<gdf_size_type>(0) + columns_to_hash[0]->size;
	auto pointer_wrapper = thrust::device_pointer_cast((unsigned long long *) output_column->data);
	thrust::transform(thrust::cuda::par.on(*stream),
			begin,
			end,
			thrust::detail::make_normal_iterator(pointer_wrapper),
			op);


	all_bitmask_on(output_column->valid, output_column->null_count, num_values,*stream);
	//and all of the bitmaps together

	for(int i = 0; i < num_columns; i++){
		if(columns_to_hash[i]->null_count > 0){
			apply_bitmask_to_bitmask( output_column->null_count,
						output_column->valid, output_column->valid,
						columns_to_hash[i]->valid, *stream, num_values);
		}
	}


	if(created_stream){
		cudaStreamSynchronize(temp_stream);
		cudaStreamDestroy(temp_stream);
	}
	cudaFree(widths);
	cudaFree(pointers);
	delete[] host_widths;
	delete[] data_holder;

	return GDF_SUCCESS;
}
