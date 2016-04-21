/*	Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/cuLDPC

    CUDA implementation of LDPC decoding algorithm.

    The details of implementation can be found from the following papers:
    1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
    2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

    The current release is close to the GlobalSIP2013 paper. 

    Created: 	10/1/2010
    Revision:	08/01/2013
                4/20/2016 prepare for release on Github.
*/

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "util/helper_cuda.h"
#include "util/timer.h"

#include "cuLDPC.h"
#include "cuLDPC_matrix.h"
#include "cuLDPC_kernel.cu"

float snr ;
long seed ;
float rate ;
int iter ;

// Extern function and variable definition
extern "C"
{
    void structure_encode (int s [], int code [], int h[BLK_ROW][BLK_COL]);
    void info_gen (int info_bin []);
    void modulation (int code [], float trans []);
    void awgn (float trans [], float recv []);
    void error_check (float trans [], float recv []);
    void llr_init (float llr [], float recv []);
    int parity_check (float app[]);
    error_result cuda_error_check (int info[], int hard_decision[]);

    float sigma ;
    int *info_bin ;
};


int printDevices();
int runTest();
int printDevices()
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("error: no devices supporting CUDA. \n");
    }
    cudaDeviceProp deviceProperty;
    int currentDeviceID = 0;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperty, currentDeviceID));

    printf("\ndevice name: %s", deviceProperty.name);
    printf("\n");
    printf("device sharedMemPerBlock: %Iu \n", deviceProperty.sharedMemPerBlock);
    printf("device totalGlobalMem: %Iu \n", deviceProperty.totalGlobalMem);
    printf("device regsPerBlock: %d \n", deviceProperty.regsPerBlock);
    printf("device warpSize: %d \n", deviceProperty.warpSize);
    printf("device memPitch: %Iu \n", deviceProperty.memPitch);
    printf("device maxThreadsPerBlock: %d \n", deviceProperty.maxThreadsPerBlock);
    printf("device maxThreadsDim[0]: %d \n", deviceProperty.maxThreadsDim[0]);
    printf("device maxThreadsDim[1]: %d \n", deviceProperty.maxThreadsDim[1]);
    printf("device maxThreadsDim[2]: %d \n", deviceProperty.maxThreadsDim[2]);
    printf("device maxGridSize[0]: %d \n", deviceProperty.maxGridSize[0]);
    printf("device maxGridSize[1]: %d \n", deviceProperty.maxGridSize[1]);
    printf("device maxGridSize[2]: %d \n", deviceProperty.maxGridSize[2]);
    printf("device totalConstMem: %Iu \n", deviceProperty.totalConstMem);
    printf("device major: %d \n", deviceProperty.major);
    printf("device minor: %d \n", deviceProperty.minor);
    printf("device clockRate: %d \n", deviceProperty.clockRate);
    printf("device textureAlignment: %Iu \n", deviceProperty.textureAlignment);
    printf("device deviceOverlap: %d \n", deviceProperty.deviceOverlap);
    printf("device multiProcessorCount: %d \n", deviceProperty.multiProcessorCount);
    printf("device zero-copy data transfers: %d \n", deviceProperty.canMapHostMemory);

    printf("\n");
    return cudaSuccess;
}


int main()
{
    printf("CUDA LDPC Decoder\r\nComputing...\r\n");
    //printDevices();
    cudaSetDevice(DEVICE_ID);
    runTest();
    return 0;
}

extern "C" int h_base [BLK_ROW][BLK_COL];

int runTest()
{
    h_element h_compact1 [H_COMPACT1_COL][H_COMPACT1_ROW]; // for update dt, R
    h_element h_element_temp;

    // init the compact matrix
    for(int i = 0; i < H_COMPACT1_COL; i++)
    {
        for(int j = 0; j < H_COMPACT1_ROW; j ++)
        {
            h_element_temp.x = 0;
            h_element_temp.y = 0;
            h_element_temp.value = -1;
            h_element_temp.valid = 0;
            h_compact1[i][j] = h_element_temp; // h[i][0-11], the same column
        }
    }

    // scan the h matrix, and gengerate compact mode of h
    for(int i = 0; i < BLK_ROW; i++)
    {
        int k=0;
        for(int j = 0; j <  BLK_COL; j ++)
        {
            if(h_base[i][j] != -1)
            {
                h_element_temp.x = i;
                h_element_temp.y = j;
                h_element_temp.value = h_base[i][j];
                h_element_temp.valid = 1;
                h_compact1[k][i] = h_element_temp;
                k++;
            }
        }
        //printf("row %d, #element=%d\n", i, k);
    }

    // h_compact2
    h_element h_compact2 [H_COMPACT2_ROW][H_COMPACT2_COL]; // for update llr

    // init the compact matrix
    for(int i = 0; i < H_COMPACT2_ROW; i++)
    {
        for(int j = 0; j < H_COMPACT2_COL; j ++)
        {
            h_element_temp.x = 0;
            h_element_temp.y = 0;
            h_element_temp.value = -1;
            h_element_temp.valid = 0;
            h_compact2[i][j] = h_element_temp;
        }
    }

    for(int j = 0; j < BLK_COL; j++)
    {
        int k=0;
        for(int i = 0; i < BLK_ROW; i ++)
        {
            if(h_base[i][j] != -1)
            {
                h_element_temp.x = i;  // although h is transposed, the (x,y) is still (iBlkRow, iBlkCol)
                h_element_temp.y = j;
                h_element_temp.value = h_base[i][j];
                h_element_temp.valid = 1;
                h_compact2[k][j] = h_element_temp;
                k++;
            }
        }
    }

    //int memorySize_h_base = BLK_ROW * BLK_COL * sizeof(int);
    int memorySize_h_compact1 = H_COMPACT1_ROW * H_COMPACT1_COL * sizeof(h_element);
    int memorySize_h_compact2 = H_COMPACT2_ROW * H_COMPACT2_COL * sizeof(h_element);

    int memorySize_infobits = INFO_LEN * sizeof(int);
    int memorySize_codeword = CODEWORD_LEN * sizeof(int);
    int memorySize_llr = CODEWORD_LEN * sizeof(float);

    int memorySize_et = MCW * CW * sizeof(int);

    info_bin = (int *) malloc(memorySize_infobits) ;
    int *codeword = (int *) malloc(memorySize_codeword) ;
    float *trans = (float *) malloc(memorySize_llr) ;
    float *recv = (float *) malloc(memorySize_llr) ;
    float *APP = (float *) malloc(memorySize_llr) ;

    float *llr = (float *) malloc(memorySize_llr) ;
    int * et = (int*) malloc(memorySize_et);

    rate = (float)0.5f;
    seed = 69012 ;
    srand (seed);

    // Create streams
    cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAMS * sizeof(cudaStream_t));
    for (int i = 0; i < NSTREAMS; i++)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    //////////////////////////////////////////////////////////////////////////////////
    // all the variables Starting with _cuda is used in host code and for cuda computation
    int memorySize_infobits_cuda = MCW * CW * memorySize_infobits ;
    int memorySize_llr_cuda = MCW *  CW * CODEWORD_LEN * sizeof(float);
    int memorySize_dt_cuda = MCW *  CW * ROW * BLK_COL * sizeof(float);
    int memorySize_R_cuda = MCW *  CW * ROW * BLK_COL * sizeof(float);
    int memorySize_hard_decision_cuda = MCW * CW * CODEWORD_LEN * sizeof(int);
    int memorySize_et_cuda = MCW * CW * sizeof(int);

    int *info_bin_cuda[NSTREAMS];
    float *llr_cuda[NSTREAMS];
    int * hard_decision_cuda[NSTREAMS];

    // Allocate pinned memory for llr and hard_decision data.
#if USE_PINNED_MEM == 1  // pinned memory
    for(int i = 0; i < NSTREAMS; i ++)
    {
        info_bin_cuda[i] = (int *) malloc(memorySize_infobits_cuda);
        checkCudaErrors(cudaHostAlloc((void **)&llr_cuda[i], memorySize_llr_cuda, cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void **)&hard_decision_cuda[i], memorySize_hard_decision_cuda, cudaHostAllocDefault));
    }
#else // pageable memory
    hard_decision_cuda = (int *) malloc(memorySize_hard_decision_cuda);
    llr_cuda = (float *) malloc(memorySize_llr_cuda);
#endif

    // create device memory
    float * dev_llr[NSTREAMS];
    float * dev_dt[NSTREAMS];
    float * dev_R[NSTREAMS];
    int * dev_hard_decision[NSTREAMS];
    int * dev_et[NSTREAMS];

    bool b_et;
    error_result this_error;

    int total_frame_error = 0;
    int total_bit_error = 0;
    int total_codeword = 0;
    int num_of_iteration_for_et = 0;
    int iter_cnt=0, iter_num =0;
    float aver_iter=0.0f; 

    for(int i = 0; i < NSTREAMS; i ++)
    {
        checkCudaErrors(cudaMalloc((void **)&dev_llr[i], memorySize_llr_cuda));
        checkCudaErrors(cudaMalloc((void **)&dev_dt[i], memorySize_dt_cuda));
        checkCudaErrors(cudaMalloc((void **)&dev_R[i], memorySize_R_cuda));
        checkCudaErrors(cudaMalloc((void **)&dev_hard_decision[i], memorySize_hard_decision_cuda));
        checkCudaErrors(cudaMalloc((void **)&dev_et[i], memorySize_et_cuda));
    }

    for(int snri = 0; snri < NUM_SNR; snri++)
    {
        snr = snr_array[snri];
        sigma = 1.0f/sqrt(2.0f*rate*pow(10.0f,(snr/10.0f)));

        total_codeword = 0;
        total_frame_error = 0;
        total_bit_error = 0;
        iter_num = 0;
        aver_iter = 0.0f;
        iter_cnt = 0;

        // In this version code, I don't care the BER performance, so don't need this loop.
        while ( (total_frame_error <= MIN_FER) && (total_codeword <= MIN_CODEWORD))
        {
            total_codeword += CW * MCW;

            for(int i = 0; i < CW * MCW; i++)
            {
                // Generating random data
                info_gen (info_bin);
                // Encoding
                structure_encode (info_bin, codeword, h_base) ;
                // BPSK modulation
                modulation (codeword, trans) ;
                // Add noise
                awgn (trans, recv) ;

#ifdef PRINT_MSG
                // Error check
                error_check (trans, recv) ;
#endif
                // LLR init
                llr_init (llr, recv) ;
                // copy the info_bin and llr to the total memory
                for(int j = 0; j < NSTREAMS; j ++)
                {
                    memcpy(info_bin_cuda[j] + i * INFO_LEN, info_bin, memorySize_infobits);
                    memcpy(llr_cuda[j] + i * CODEWORD_LEN, llr, memorySize_llr);
                }
            }

#if MEASURE_CUDA_TIME == 1
            // start the timer
            cudaEvent_t start_kernel, stop_kernel, start_h2d, stop_h2d, start_d2h, stop_d2h;
            cudaEvent_t start_memset, stop_memset;
            cudaEventCreate(&start_kernel);
            cudaEventCreate(&stop_kernel);
            cudaEventCreate(&start_h2d);
            cudaEventCreate(&stop_h2d);
            cudaEventCreate(&start_d2h);
            cudaEventCreate(&stop_d2h);
            cudaEventCreate(&start_memset);
            cudaEventCreate(&stop_memset);

            float time_kernel = 0.0, time_kernel_temp = 0.0;
            float time_h2d=0.0, time_h2d_temp = 0.0;
            float time_d2h=0.0, time_d2h_temp = 0.0;
            float time_memset = 0.0f, time_memset_temp = 0.0;
#endif

            // Since for all the simulation, this part only transfer once. 
            // So the time we don't count into the total time.
            checkCudaErrors(cudaMemcpyToSymbol(dev_h_compact1, h_compact1, memorySize_h_compact1));  // constant memory init.
            checkCudaErrors(cudaMemcpyToSymbol(dev_h_compact2, h_compact2, memorySize_h_compact2));  // constant memory init.
            int blockSizeX = (Z + 32 - 1)/ 32 * 32;

            // Define CUDA kernel dimension
            dim3 dimGridKernel1(BLK_ROW, MCW, 1); // dim of the thread blocks
            dim3 dimBlockKernel1(blockSizeX, CW, 1);
            int threadsPerBlockKernel1 = blockSizeX * CW;
            int sharedRCacheSize = threadsPerBlockKernel1 * NON_EMPTY_ELMENT * sizeof(float);

            dim3 dimGridKernel2(BLK_COL, MCW, 1);
            dim3 dimBlockKernel2(blockSizeX, CW, 1);
            int threadsPerBlockKernel2 = blockSizeX * CW;
            int sharedDtCacheSize = threadsPerBlockKernel2 * NON_EMPTY_ELMENT_VNP * sizeof(float);

            dim3 dimGridKernel3(MCW, CW, 1);
            dim3 dimBlockKernel3(Z, BLK_ROW, 1);

#if MEASURE_CPU_TIME == 1
            // cpu timer
            float cpu_run_time = 0.0;
            Timer cpu_timer;
            cpu_timer.start();	
#endif

            // run the kernel
            for(int j = 0; j < MAX_SIM; j++)
            {
#if MEASURE_CUDA_TIME == 1
                cudaEventRecord(start_h2d,0);
                //cudaEventSynchronize(start_h2d);
#endif

                // Transfer LLR data into device.
#if USE_PINNED_MEM == 1
                for(int iSt = 0; iSt < NSTREAMS; iSt ++)
                {
                    checkCudaErrors(cudaMemcpyAsync(dev_llr[iSt], llr_cuda[iSt], memorySize_llr_cuda, cudaMemcpyHostToDevice, streams[iSt]));
                    cudaStreamSynchronize(streams[iSt]);
                }
                //cudaDeviceSynchronize();
#else
                checkCudaErrors(cudaMemcpy(dev_llr, llr_cuda, memorySize_llr_cuda, cudaMemcpyHostToDevice));
#endif

#if MEASURE_CUDA_TIME == 1
                cudaEventRecord(stop_h2d, 0);
                cudaEventSynchronize(stop_h2d);
                cudaEventElapsedTime(&time_h2d_temp, start_h2d, stop_h2d);
                time_h2d+=time_h2d_temp;
#endif

#if MEASURE_CUDA_TIME == 1
                cudaEventRecord(start_memset,0);
#endif

#if ETA == 1
                checkCudaErrors(cudaMemset(dev_et, 0, memorySize_et_cuda));
#endif

#if MEASURE_CUDA_TIME == 1
                cudaEventRecord(stop_memset,0);
                cudaEventSynchronize(stop_memset);
                cudaEventElapsedTime(&time_memset_temp, start_memset, stop_memset);
                time_memset += time_memset_temp;
#endif

                for(int iSt = 0; iSt < NSTREAMS; iSt ++)
                {
                    checkCudaErrors(cudaMemcpyAsync(dev_llr[iSt], llr_cuda[iSt], memorySize_llr_cuda, cudaMemcpyHostToDevice, streams[iSt]));

                    // kernel launch
                    for(int ii = 0; ii < MAX_ITERATION; ii++)
                    {
                        if(ii == 0)
                            ldpc_cnp_kernel_1st_iter<<<dimGridKernel1,dimBlockKernel1, 0, streams[iSt]>>>(dev_llr[iSt], dev_dt[iSt], dev_R[iSt], dev_et[iSt]);
                        else
                            ldpc_cnp_kernel<<<dimGridKernel1,dimBlockKernel1, sharedRCacheSize, streams[iSt]>>>(dev_llr[iSt], dev_dt[iSt], dev_R[iSt], dev_et[iSt], threadsPerBlockKernel1);

                        if(ii < MAX_ITERATION - 1)
                            ldpc_vnp_kernel_normal<<<dimGridKernel2,dimBlockKernel2, 0, streams[iSt]>>>(dev_llr[iSt], dev_dt[iSt], dev_et[iSt]);
                        else
                            ldpc_vnp_kernel_last_iter<<<dimGridKernel2,dimBlockKernel2, 0, streams[iSt]>>>(dev_llr[iSt], dev_dt[iSt], dev_hard_decision[iSt], dev_et[iSt]);
                    }

                    checkCudaErrors(cudaMemcpyAsync(hard_decision_cuda[iSt], dev_hard_decision[iSt], memorySize_hard_decision_cuda, cudaMemcpyDeviceToHost, streams[iSt]));


                    num_of_iteration_for_et = MAX_ITERATION;
                }

                cudaDeviceSynchronize();

#if MEASURE_CUDA_TIME == 1
                cudaEventRecord(stop_d2h, 0);
                cudaEventSynchronize(stop_d2h);
                cudaEventElapsedTime(&time_d2h_temp, start_d2h, stop_d2h);
                time_d2h+=time_d2h_temp;
#endif

#ifdef DISPLAY_BER
                for(int iSt = 0; iSt < NSTREAMS; iSt ++)
                {
                    this_error = cuda_error_check(info_bin_cuda[iSt], hard_decision_cuda[iSt]);
                    total_bit_error += this_error.bit_error;
                    total_frame_error += this_error.frame_error;
                }
#endif

#if ETA == 1
                iter_num += num_of_iteration_for_et;
                iter_cnt ++;
                aver_iter = (float)iter_num * 1.0f / iter_cnt;
#endif
            } // end of MAX-SIM
        } // end of the MAX frame error.


#if MEASURE_CUDA_TIME == 1
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
#endif

#if MEASURE_CPU_TIME == 1
        cudaDeviceSynchronize();
        cpu_timer.stop();
        cpu_run_time += cpu_timer.stop_get();

        printf ("\n=================================\n\r");
        printf ("GPU CUDA Demo\n");
        printf ("SNR = %1.1f dB\n", snr);
        printf ("# codewords = %d, # streams = %d, CW=%d, MCW=%d\r\n",total_codeword * NSTREAMS, NSTREAMS, CW, MCW);
        printf("number of iterations = %1.1f \r\n", aver_iter);
        printf("CPU time: %f ms, for %d simulations.\n", cpu_run_time, MAX_SIM);
        float throughput = (float)CODEWORD_LEN * NSTREAMS * MCW * CW * MAX_SIM / cpu_run_time /1000;
        printf("Throughput = %f Mbps\r\n", (float)CODEWORD_LEN * NSTREAMS * MCW * CW * MAX_SIM / cpu_run_time /1000);			
#endif

#if MEASURE_CUDA_TIME == 1
        printf("Throughput (kernel only) = %f Mbps\r\n", (float)CODEWORD_LEN * MCW * CW * MAX_SIM / time_kernel /1000);
        printf("Throughput (kernel + transer time) = %f Mbps\r\n", (float)CODEWORD_LEN * MCW * CW * MAX_SIM / (time_kernel + time_h2d+ time_d2h + time_memset)  /1000);
        float bandwidthInMBs = (1e3f * memorySize_llr_cuda ) /  ( (time_h2d/MAX_SIM) * (float)(1 << 20));
        printf("\nh2d (llr): size=%f MB, bandwidthInMBs = %f MB/s\n", memorySize_llr_cuda /1e6, bandwidthInMBs);
        bandwidthInMBs = (1e3f *  memorySize_hard_decision_cuda) /  ( (time_d2h/MAX_SIM) * (float)(1 << 20));
        printf("d2h (hd): size=%f MB, bandwidthInMBs = %f MB/s\n", memorySize_hard_decision_cuda /1e6, bandwidthInMBs);

        printf ("kernel time = %f ms \nh2d time = %f ms \nd2h time = %f ms\n", time_kernel, time_h2d, time_d2h);
        printf ("memset time = %f ms \n", time_memset);
        printf ("time difference = %f ms \n", cpu_run_time - time_kernel - time_h2d - time_d2h - time_memset);
#endif

#ifdef DISPLAY_BER
        printf ("# codewords = %d, CW=%d, MCW=%d\r\n",total_codeword, CW, MCW);
        printf ("total bit error = %d\n", total_bit_error);
        printf ("BER = %1.2e, FER = %1.2e\n", (float)total_bit_error/total_codeword/INFO_LEN, (float)total_frame_error/total_codeword);
#endif
    }// end of the snr loop

    for(int iSt = 0; iSt < NSTREAMS; iSt ++)
    {
        checkCudaErrors(cudaFree(dev_llr[iSt]));
        checkCudaErrors(cudaFree(dev_dt[iSt]));
        checkCudaErrors(cudaFree(dev_R[iSt]));
        checkCudaErrors(cudaFree(dev_hard_decision[iSt]));
        checkCudaErrors(cudaFree(dev_et[iSt]));
        free(info_bin_cuda[iSt]);
        checkCudaErrors(cudaStreamDestroy(streams[iSt]));
    }

    free(info_bin);
    free(codeword);
    free(trans);
    free(recv);
    free(llr);
    free(et);

#if USE_PINNED_MEM == 1
    for(int iSt = 0; iSt < NSTREAMS; iSt ++)
    {
        checkCudaErrors(cudaFreeHost(llr_cuda[iSt]));
        checkCudaErrors(cudaFreeHost(hard_decision_cuda[iSt]));
    }
#else
    free(llr_cuda);
    free(hard_decision_cuda);
#endif

    return 0;
}


