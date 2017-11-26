/*	Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/cuLDPC

    CUDA implementation of LDPC decoding algorithm.

    The details of implementation can be found from the following papers:
    1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
    2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

    The current release is close to the GlobalSIP2013 paper.

    Created: 	10/1/2010
    Revision:	08/01/2013
              04/20/2016 prepare for release on Github.
              11/26/2017 cleanup and comments by Albin Severinson, albin (at) severinson.org
*/

#ifndef LDPC_CUDA_KERNEL_CU
#define LDPC_CUDA_KERNEL_CU

#include <stdio.h>
#include "cuLDPC.h"

// constant memory
//__device__ __constant__ int  dev_h_base[H_MATRIX];
__device__ __constant__ h_element dev_h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW];  // used in kernel 1
__device__ __constant__ h_element dev_h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL];  // used in kernel 2

// For cnp kernel
#if MODE == WIMAX
__device__ __constant__ char h_element_count1[BLK_ROW] = {6, 7, 7, 6, 6, 7, 6, 6, 7, 6, 6, 6};
__device__ __constant__ char h_element_count2[BLK_COL] = {3, 3, 6, 3, 3, 6, 3, 6, 3, 6, 3, 6, \
                                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#else
__device__ __constant__ char h_element_count1[BLK_ROW] = {7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 8};
__device__ __constant__ char h_element_count2[BLK_COL] = {11,4, 3, 3,11, 3, 3, 3,11, 3, 3, 3, \
                                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#endif

// MIN-SUM Function. The compiler automatically inlines __device__ functions in
// most cases.
__device__ float F_FUCN_MIN_SUM_DEV(float a, float b)
{
  float a_abs;
  float b_abs;
  float min_ab_abs;
  float sign_a;
  float sign_b;
  float tmp;

  // get the sign of a and b
  sign_a = a < 0 ? -1.0f : 1.0f;
  sign_b = b < 0 ? -1.0f : 1.0f;

  // device float multiplication with rounding to nearest integer.
  // TODO: Is this method faster than dedicated abs function?
  a_abs = __fmul_rn(sign_a, a);
  b_abs = __fmul_rn(sign_b, b);

  min_ab_abs = fmin(a_abs, b_abs);
  tmp = sign_a * sign_b * min_ab_abs;
  return tmp;
}

// Kernel 1
__global__ void ldpc_cnp_kernel_1st_iter(float * dev_llr,
                                         float * dev_dt,
                                         float * dev_R,
                                         int * dev_et)
{
#if MODE == WIFI
  if(threadIdx.x >= Z)
    return;
#endif

  int iCW = threadIdx.y; // index of CW in a MCW
  int iMCW = blockIdx.y; // index of MCW
  int iCurrentCW = iMCW * CW + iCW;

#if ET_MARK == 1
  if(dev_et[iCurrentCW] == 1)
    return;
#endif

  //for step 1: update dt
  int iBlkRow; // block row in h_base
  int iBlkCol; // block col in h_base
  int iSubRow; // row index in sub_block of h_base
  int iCol; // overall col index in h_base
  int offsetR;

  iSubRow = threadIdx.x;
  iBlkRow = blockIdx.x;

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block
  int shift_t;

  // For 2-min algorithm.
  char Q_sign = 0;
  char sq;
  float Q, Q_abs;
  float R_temp;

  float sign = 1.0f;
  float rmin1 = 1000.0f;
  float rmin2 = 1000.0f;
  char idx_min = 0;

  h_element h_element_t;
  int s = h_element_count1[iBlkRow];
  offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

  // The 1st recursion
  for(int i = 0; i < s; i++) // loop through all the ZxZ sub-blocks in a row
    {
      h_element_t = dev_h_compact1[i][iBlkRow];

      iBlkCol = h_element_t.y;
      shift_t = h_element_t.value;

      shift_t = (iSubRow + shift_t);
      if(shift_t >= Z) shift_t = shift_t - Z;

      iCol = iBlkCol * Z + shift_t;

      Q = dev_llr[size_llr_CW * iCurrentCW + iCol];// - R_temp;
      Q_abs = fabsf(Q);
      sq = Q < 0;

      // quick version
      sign = sign * (1 - sq * 2);
      Q_sign |= sq << i;

      if (Q_abs < rmin1)
        {
          rmin2 = rmin1;
          rmin1 = Q_abs;
          idx_min = i;
        } else if (Q_abs < rmin2)
        {
          rmin2 = Q_abs;
        }
    }

  // The 2nd recursion
  for(int i = 0; i < s; i ++)
    {
      // v0: Best performance so far. 0.75f is the value of alpha.
      sq = 1 - 2 * ((Q_sign >> i) & 0x01);
      R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

      // write results to global memory
      h_element_t = dev_h_compact1[i][iBlkRow];
      int addr_temp = offsetR + h_element_t.y * ROW;
      dev_dt[addr_temp] = R_temp;// - R1[i]; // compute the dt value for current llr.
      dev_R[addr_temp] = R_temp; // update R, R=R'.
    }
}

// Kernel_1
__global__ void ldpc_cnp_kernel(float * dev_llr,
                                float * dev_dt,
                                float * dev_R,
                                int * dev_et,
                                int threadsPerBlock)
{
#if MODE == WIFI
  if(threadIdx.x >= Z)
    return;
#endif

  // Define cache for R: Rcache[NON_EMPTY_ELMENT][nThreadPerBlock]
  // extern means that the memory is allocated dynamically at run-time
  extern __shared__ float RCache[];
  int iRCacheLine = threadIdx.y * blockDim.x + threadIdx.x;

  int iCW = threadIdx.y; // index of CW in a MCW
  int iMCW = blockIdx.y; // index of MCW
  int iCurrentCW = iMCW * CW + iCW;

#if ET_MARK == 1
  if(dev_et[iCurrentCW] == 1)
    return;
#endif

  //for step 1: update dt
  int iBlkRow; // block row in h_base
  int iBlkCol; // block col in h_base
  int iSubRow; // row index in sub_block of h_base
  int iCol; // overall col index in h_base
  int offsetR;

  iSubRow = threadIdx.x;
  iBlkRow = blockIdx.x;

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

  //float R1[NON_EMPTY_ELMENT];
  int shift_t;

  // For 2-min algorithm.
  char Q_sign = 0;
  char sq;
  float Q, Q_abs;
  float R_temp;

  float sign = 1.0f;
  float rmin1 = 1000.0f;
  float rmin2 = 1000.0f;
  char idx_min = 0;

  h_element h_element_t;
  int s = h_element_count1[iBlkRow];
  offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

  // The 1st recursion
  // TODO: Is s always the same? If so we can unroll the loop with #pragma unroll
  for(int i = 0; i < s; i++) // loop through all the ZxZ sub-blocks in a row
    {
      h_element_t = dev_h_compact1[i][iBlkRow];

      iBlkCol = h_element_t.y;
      shift_t = h_element_t.value;

      shift_t = (iSubRow + shift_t);
      if(shift_t >= Z) shift_t = shift_t - Z;

      iCol = iBlkCol * Z + shift_t;

      R_temp = dev_R[offsetR + iBlkCol * ROW];

      RCache[i * threadsPerBlock + iRCacheLine] =  R_temp;

      Q = dev_llr[size_llr_CW * iCurrentCW + iCol] - R_temp;
      Q_abs = fabsf(Q);

      sq = Q < 0;
      sign = sign * (1 - sq * 2);
      Q_sign |= sq << i;

      if (Q_abs < rmin1)
        {
          rmin2 = rmin1;
          rmin1 = Q_abs;
          idx_min = i;
        } else if (Q_abs < rmin2)
        {
          rmin2 = Q_abs;
        }
    }

  // The 2nd recursion
  //#pragma unroll
  for(int i = 0; i < s; i ++)
    {
      sq = 1 - 2 * ((Q_sign >> i) & 0x01);
      R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

      // write results to global memory
      h_element_t = dev_h_compact1[i][iBlkRow];
      int addr_temp = h_element_t.y * ROW + offsetR;
      dev_dt[addr_temp] = R_temp - RCache[i * threadsPerBlock + iRCacheLine];
      dev_R[addr_temp] = R_temp; // update R, R=R'.
    }
}

// Kernel 2: VNP processing
__global__ void
ldpc_vnp_kernel_normal(float * dev_llr, float * dev_dt, int * dev_et)
{

#if MODE == WIFI
	if(threadIdx.x >= Z)
		return;
#endif

	int	iCW = threadIdx.y; // index of CW in a MCW
	int iMCW = blockIdx.y; // index of MCW
	int iCurrentCW = iMCW * CW + iCW;

#if ET_MARK == 1
	if(dev_et[iCurrentCW] == 1)
		return;
#endif

	int iBlkCol;
	int iBlkRow;
	int iSubCol;
	int iRow;
	int iCol;

	int shift_t, sf;
	int llr_index;
	float APP;

	h_element h_element_t;

	iBlkCol = blockIdx.x;
	iSubCol = threadIdx.x;

	int size_llr_CW = COL; // size of one llr CW block
	int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

	// update all the llr values
	iCol = iBlkCol * Z + iSubCol;
	llr_index = size_llr_CW * iCurrentCW + iCol;

	APP = dev_llr[llr_index];
	int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

	for(int i = 0; i < h_element_count2[iBlkCol]; i++)
	{
		h_element_t = dev_h_compact2[i][iBlkCol];

		shift_t = h_element_t.value;
		iBlkRow = h_element_t.x;

		sf = iSubCol - shift_t;
		if(sf < 0) sf = sf + Z;

		iRow = iBlkRow * Z + sf;
		APP = APP + dev_dt[offsetDt + iRow];
	}
	// write back to device global memory
	dev_llr[llr_index] = APP;

	// No hard decision for non-last iteration.
}

// Kernel: VNP processing for the last iteration.
__global__ void ldpc_vnp_kernel_last_iter(float * dev_llr,
                                          float * dev_dt,
                                          int * dev_hd,
                                          int * dev_et)
{
#if MODE == WIFI
  if(threadIdx.x >= Z)
    return;
#endif

  int	iCW = threadIdx.y; // index of CW in a MCW
  int iMCW = blockIdx.y; // index of MCW
  int iCurrentCW = iMCW * CW + iCW;

#if ET_MARK == 1
  if(dev_et[iCurrentCW] == 1)
    return;
#endif

  int iBlkCol;
  int iBlkRow;
  int iSubCol;
  int iRow;
  int iCol;

  int shift_t, sf;
  int llr_index;
  float APP;

  h_element h_element_t;

  iBlkCol = blockIdx.x;
  iSubCol = threadIdx.x;

  int size_llr_CW = COL; // size of one llr CW block
  int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

  // update all the llr values
  iCol = iBlkCol * Z + iSubCol;
  llr_index = size_llr_CW * iCurrentCW + iCol;

  APP = dev_llr[llr_index];

  int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

  for(int i = 0; i < h_element_count2[iBlkCol]; i ++)
    {
      h_element_t = dev_h_compact2[i][iBlkCol];

      shift_t = h_element_t.value;
      iBlkRow = h_element_t.x;

      sf = iSubCol - shift_t;
      if(sf < 0) sf = sf + Z;

      iRow = iBlkRow * Z + sf;
      APP = APP + dev_dt[offsetDt + iRow];
    }

  // For the last iteration, we don't need to write intermediate results to
  // global memory. Instead, we directly make a hard decision.
  if(APP > 0)
    dev_hd[llr_index] = 0;
  else
    dev_hd[llr_index] = 1;
}

__device__ int shared_hd[CODEWORD_LEN];

// Kernel 3: Early termination.
__global__ void ldpc_decoder_kernel3_early_termination(int * dev_hd, int * dev_et)
{
  int iMCW = blockIdx.x; // index of MCW
  int iCW = blockIdx.y; // index of CW in a MCW
  int iCodeword = iMCW * CW + iCW;

#if ET_MARK == 1
  if(dev_et[iCodeword] == 1)
    return;
#endif

  //int iBlkRow = threadIdx.x;
  //int iSubRow = threadIdx.y;
  int iBlkRow = threadIdx.y;
  int iSubRow = threadIdx.x;
  int iRow = iBlkRow * Z + iSubRow;
  int tid = iRow;
  int iCol;
  int iBlkCol;

  h_element h_element_t;
  int shift_t;
  int et_result_per_row = 0;

  __shared__ h_element shared_h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW];
  __shared__ int shared_et_per_codeword;

  if(threadIdx.x == 0 && threadIdx.y == 0)
    shared_et_per_codeword = 1;
  __syncthreads();

  // read all the data into shared memory
  int hd_start_addr = iCodeword * CODEWORD_LEN;

  // read the hard decision value into the shared memory
  // every thread should read two hard decision value. Since we have 972 threads, for 1944 bits
  shared_hd[iRow] = dev_hd[hd_start_addr + iRow];
  shared_hd[iRow + INFO_LEN] = dev_hd[hd_start_addr + iRow + INFO_LEN];

  // read the h_matrix into the shared memory
  if(tid < H_COMPACT1)
    *(*shared_h_compact1 + tid) = *(*dev_h_compact1 + tid);

  __syncthreads();

  for(int i = 0; i < H_COMPACT1_COL; i++)
    {
      h_element_t = shared_h_compact1[i][iBlkRow];
      if(h_element_t.valid == 0)
        break;

      iBlkCol = h_element_t.y;

      shift_t = h_element_t.value;
      shift_t = (iSubRow + shift_t);
      if(shift_t >= Z) shift_t = shift_t - Z;

      iCol = iBlkCol * Z + shift_t;

      et_result_per_row = et_result_per_row ^ shared_hd[iCol];
    }

  // Reduction
  if(et_result_per_row == 1)
    shared_et_per_codeword = 0;

  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0)
    dev_et[iCodeword] = shared_et_per_codeword;
}

#endif
