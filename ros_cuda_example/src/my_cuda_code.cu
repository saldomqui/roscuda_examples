#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void random_init(curandState_t *states, unsigned long long rnd_seed, const int matrixDim){
    int idx = threadkIdx.x + blockIdx.x * blockDim.x;

    if (idx < matrixDim)
        curand_init(rnd_seed,(unsigned long long)idx,0,&states[idx]);
    }
}

__global__ void random_data(curandState_t *states, double *data, const int matrixDim){
    int idx = threadkIdx.x + blockIdx.x * blockDim.x;

    if (idx < matrixDim)
            curandState_t local_state;

            local_state=states[idx];
            delta[idx]=(double)(100.0*(curand_uniform(&local_state)));
    }
}

__global__ void process_data(double *data_in, double *data_out, const int matrixDim)
{
    int idx = threadkIdx.x + blockIdx.x * blockDim.x;

    if (idx < matrixDim)
    {
        double value = data_in[idx];
        data_out[idx] = value * value;

        __syncthreads(); // After this line the shared variables matchTF and numPtsPerThread necessary for next step should be filled up
    }
}
