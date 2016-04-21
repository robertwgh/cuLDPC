/*	Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
	All rights reserved. https://github.com/robertwgh/cuLDPC
	
	CUDA implementation of LDPC decoding algorithm.
	Created: 	10/1/2010
	Revision:	08/01/2013
				/4/20/2016 prepare for release on Github.
*/

#ifndef LDPC_H
#define LDPC_H

#define YES	1
#define NO	0

#define DEVICE_ID	3

// LDPC decoder configurations
#define WIMAX	0
#define WIFI	1
#define MODE	WIMAX
#define MIN_SUM	YES		//otherwise, log-SPA

// Simulation parameters
#define NUM_SNR 1
static float snr_array[NUM_SNR] = {3.0f};
#define MIN_FER         2000000
#define MIN_CODEWORD    9000000
#define MAX_ITERATION 5
#define DEBUG_BER	NO

// Number of streams
#define NSTREAMS 1
#define CW 10
#define MCW 100
#define MAX_SIM 500

#define MEASURE_CPU_TIME	YES		// whether measure time and throughput
#define MEASURE_CUDA_TIME	NO		// whether measure CUDA memory transfer time and CUDA kernel time
//#	define DISPLAY_BER	

// begin debug
#define	DEBUG_FILE	NO
#define PRINT_MSG	NO
// end debug

// Performance optimizations
#define USE_PINNED_MEM	YES
#define ETA				NO		//early termination algorithm
#define ET_MARK			NO

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//	The following settings are fixed.
//	They don't need to be changed during simulations.
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#if MODE == WIMAX
// WIMAX
#define Z				96 //1024//96
#define NON_EMPTY_ELMENT 7
#define NON_EMPTY_ELMENT_VNP	6
#else 
// 802.11n
#define Z				81
#define NON_EMPTY_ELMENT 8 //the maximum number of non-empty element in a row of H matrix, for 1944 bit 802.11n code, NON_EMPTY_ELMENT=8
#define NON_EMPTY_ELMENT_VNP	11
#endif

#define BLK_ROW			12
#define BLK_COL			24
#define HALF_BLK_COL	12

#define BLK_INFO		BLK_ROW
#define BLK_CODEWORD	BLK_COL

#define ROW				(Z*BLK_ROW)
#define COL				(Z*BLK_COL)
#define INFO_LEN		(BLK_INFO * Z)
#define CODEWORD_LEN	(BLK_CODEWORD * Z)

// the slots in the H matrix
#define H_MATRIX		288
// the slots in the compact H matrix
#define H_COMPACT1_ROW	BLK_ROW
#define H_COMPACT1_COL	NON_EMPTY_ELMENT
#define H_COMPACT1 (BLK_ROW * NON_EMPTY_ELMENT) //96 // 8*12

#define H_COMPACT2_ROW	BLK_ROW
#define H_COMPACT2_COL	BLK_COL

typedef struct
{
	int bit_error;
	int frame_error;
} error_result;

typedef struct
{
	char x;
	char y;
	char value;
	char valid;
} h_element;

#endif
