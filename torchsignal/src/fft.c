#include <TH/TH.h>
#include <fftw3.h>

// Adapted from 
// https://github.com/mbhenaff/spectral-lib/blob/master/cuda/cufft.cpp
// http://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html 
// https://github.com/koraykv/torch-fftw/tree/master/lib/thfftw
int fft1_c2c(THFloatTensor *input, THFloatTensor *output, int dir)
{
  long nInputLines = input->size[0];
  long N = input->size[1];
  int size[2] = {N};

  THFloatTensor_resizeAs(output, input);

  THArgCheck(THFloatTensor_isContiguous(input), 2, "Input tensor must be contiguous");
  THArgCheck(THFloatTensor_isContiguous(output), 2, "Output tensor must be contiguous");

  // raw pointers 
  fftwf_complex *input_data = (fftwf_complex*)THFloatTensor_data(input);
  fftwf_complex *output_data = (fftwf_complex*)THFloatTensor_data(output);
	
  fftwf_plan plan = fftwf_plan_many_dft(
      1, size, nInputLines, 
      input_data, size, 1, N, 
      output_data, size, 1, N, 
      -dir, FFTW_ESTIMATE); 
  fftwf_execute(plan);

  // clean up
  fftwf_destroy_plan(plan);
  return 0;	
}

int fft2_c2c(THFloatTensor *input, THFloatTensor *output, int dir)
{
  long nInputPlanes = input->size[0];
  long N = input->size[1];
  long M = input->size[2];
  int size[2] = {N, M};

  THFloatTensor_resizeAs(output, input);

  THArgCheck(THFloatTensor_isContiguous(input), 2, "Input tensor must be contiguous");
  THArgCheck(THFloatTensor_isContiguous(output), 2, "Output tensor must be contiguous");

  // raw pointers 
  fftwf_complex *input_data = (fftwf_complex*)THFloatTensor_data(input);
  fftwf_complex *output_data = (fftwf_complex*)THFloatTensor_data(output);
	
  // execute FFT
  // dir: CUFFT_FORWARD, CUFFT_INVERSE

  fftwf_plan plan = fftwf_plan_many_dft(
      2, size, nInputPlanes, 
      input_data, size, 1, M*N, 
      output_data, size, 1, M*N, 
      -dir, FFTW_ESTIMATE); 
  fftwf_execute(plan);

  // clean up
  fftwf_destroy_plan(plan);
  return 0;	
}

int fft3_c2c(THFloatTensor *input, THFloatTensor *output, int dir)
{
  long nInputPlanes = input->size[0];
  long N = input->size[1];
  long M = input->size[2];
  long K = input->size[3];
  int size[3] = {N, M, K};

  THFloatTensor_resizeAs(output, input);

  THArgCheck(THFloatTensor_isContiguous(input), 2, "Input tensor must be contiguous");
  THArgCheck(THFloatTensor_isContiguous(output), 2, "Output tensor must be contiguous");

  // raw pointers 
  fftwf_complex *input_data = (fftwf_complex*)THFloatTensor_data(input);
  fftwf_complex *output_data = (fftwf_complex*)THFloatTensor_data(output);
	
  // execute FFT
  // dir: CUFFT_FORWARD, CUFFT_INVERSE

  fftwf_plan plan = fftwf_plan_many_dft(
      3, size, nInputPlanes, 
      input_data, size, 1, M*N*K, 
      output_data, size, 1, M*N*K, 
      -dir, FFTW_ESTIMATE); 
  fftwf_execute(plan);

  // clean up
  fftwf_destroy_plan(plan);
  return 0;	
}
