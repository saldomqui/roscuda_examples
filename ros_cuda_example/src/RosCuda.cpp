
#include "RosCuda.h"

#define NUM_THREADS_PER_BLOCK 100 // Less than 1024

using namespace std;

// RosCuda class constructor
RosCuda::RosCuda() : local_nh_("~"),
                     global_nh_("")
{
    //-------------------------------------------- Global parameters -------------------------------------------

    //-------------------------------------------- Local parameters -------------------------------------------

    //------------------------------- Declare any publishers, subscribers or services here -----------------------
    dim_sub = global_nh_.subscribe<std_msgs::Int32>("dimension", 10, &RosCuda::DimensionCallback, this);
}

// Destructor
RosCuda::~RosCuda()
{
    ros::waitForShutdown();
}

// Callback of the matrix dimension
void RosCuda::DimensionCallback(const std_msgs::Int32 msg)
{
    int threadsPerBlock;
    int numCudaBlocks;
    double matrixDimension;
    int matrixSize;
    int dataSize;
    double *h_data;
#if __aarch64__
    curandState_t *d_states;
    double *d_data_in;
    double *d_data_out;
#endif

    matrixDimension = msg.data;

    cout << "RosCuda: Processing matrix of dimension:" << matrixDimension << endl;

    // Compute sizes
    matrixSize = matrixDimension * matrixDimension;
    dataSize = matrixSize * sizeof(double);

    //------------ START COUNTING PROCESSING TIME -------------------------
    auto begin = std::chrono::high_resolution_clock::now();

    //---------------- ALLOCATE MEMORYT ------------------------------------------
    h_data = (double *)malloc(dataSize*sizeof(double));
#if __aarch64__
    cudaMalloc((void **)&d_states,matrixDimension*sizeof(curandState_t));
    cudaMalloc((void **)&d_data_in, dataSize);
    cudaMalloc((void **)&d_data_out, dataSize);
#endif

#if __aarch64__

    threadsPerBlock = NUM_THREADS_PER_BLOCK;
    numCudaBlocks = ceil(matrixSize / threadsPerBlock);

    cout << "RosCuda: numCudaBlocks:" << numCudaBlocks << " threadsPerBlock:" << threadsPerBlock << endl;

    //--------- GENERATE RANDOM DATA  [0,100] ------------------------------------
    random_init<<<numCudaBlocks,threadsPerBlock>>>(d_states,static_cast<unsigned long long>(time(0)),matrixDimension);
    random_data(d_states, data_in, matrixDimension);


    /*
    // Here copy data from host to on device
    //cudaMemcpy(d_data_in, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
*/

    //-------------- COMPUTE DATA -----------------------------------------------
    process_data<<<numCudaBlocks, threadsPerBlock>>>(d_data_in, d_data_out, matrixDimension);
    /*
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Processing time:" << elapsedTime << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
*/
    //------------- GET RESULT FROM DEVICE ---------------------------------------
    cudaMemcpy(h_data, d_data_out, data_Size, cudaMemcpyDeviceToHost);

    //------------ FREE MEMORY -------------------------
    cudaFree(d_data_in);
    cudaFree(d_data_out);
    cudaFree(d_states);

#else // Do it with CPU
    //--------- GENERATE RANDOM DATA  [0,100] ------------------------------------
    for (int i = 0; i < matrixDimension; i++)
        for (int j = 0; j < matrixDimension; j++)
            h_data[static_cast<unsigned long>(i+j*matrixDimension)] = ((double)rand() / RAND_MAX) * 100;

    //-------------- COMPUTE DATA -----------------------------------------------
    for (int i = 0; i < matrixDimension; i++)
        for (int j = 0; j < matrixDimension; j++){
            h_data[static_cast<unsigned long>(i+j*matrixDimension)] *= h_data[static_cast<unsigned long>(i+j*matrixDimension)];
        }
#endif
    free(h_data);

     // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
}
