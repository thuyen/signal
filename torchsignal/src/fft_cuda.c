#include <THC/THC.h>
#include <TH/TH.h>
#include <cufft.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Adapted from 
// https://github.com/mbhenaff/spectral-lib/blob/master/cuda/cufft.cpp
int fft1_c2c_cuda(THCudaTensor *input, THCudaTensor *output, int dir)
{
  long nInputLines = input->size[0];
  long N = input->size[1];

  THCudaTensor_resizeAs(state, output, input);

  THArgCheck(THCudaTensor_isContiguous(NULL, input), 2, "Input tensor must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(NULL, output), 2, "Output tensor must be contiguous");

  // raw pointers 
  cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL, input);
  cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL, output);
	
  // execute FFT
  // dir: CUFFT_FORWARD, CUFFT_INVERSE
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, nInputLines);
  cufftExecC2C(plan, (cufftComplex*)input_data, (cufftComplex*)output_data, -dir);

  // clean up
  cufftDestroy(plan);
  return 0;	
}

int fft2_c2c_cuda(THCudaTensor *input, THCudaTensor *output, int dir)
{
  long nInputPlanes = input->size[0];
  long N = input->size[1];
  long M = input->size[2];
  int size[2] = {N, M};

  THCudaTensor_resizeAs(state, output, input);

  THArgCheck(THCudaTensor_isContiguous(NULL, input), 2, "Input tensor must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(NULL, output), 2, "Output tensor must be contiguous");

  // raw pointers 
  cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL, input);
  cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL, output);

  // execute FFT
  // dir: CUFFT_FORWARD, CUFFT_INVERSE
  cufftHandle plan;
  cufftPlanMany(&plan, 2, size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nInputPlanes);
  cufftExecC2C(plan, (cufftComplex*)input_data, (cufftComplex*)output_data, -dir);
  return 0;
}

