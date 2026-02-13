#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "gptq_kernel.h"

#define BLOCK_SIZE 64   // GPTQ quantization block size
#define TILE_WIDTH 32   // Step 2 update tile width

__device__ __forceinline__ float quantize_val(
        float w,
        float scale,
        float zero,
        float qmin,
        float qmax
        ) {
    float q = rintf(w / scale + zero);
    q = fmaxf(q, qmin);
    q = fminf(q, qmax);
    return (q - zero) * scale;
}


__global__ void gptq_fused_kernel(
        float* __restrict__ W,          // M by N
        float* __restrict__ Err,        // M by BLOCK_SIZE
        const float* __restrict__ H,    // N by N
        const float* __restrict__ Scales,
        const float* __restrict__ Zeros,
        int M, int N,
        int col_offset,
        float qmin, float qmax
        ) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Share memory for H block (step 1) and H tile (step2)
    extern __shared__ float sh_mem[];
    float* sh_H = sh_mem;   // BLOCK_SIZE by BLOCK_SIZE in step 1
                            // BLOCK_SIZE by TILE_WIDTH in step 2
    float* sh_inv_diag = &sh_mem[BLOCK_SIZE * BLOCK_SIZE];
    // load H[offset:offset + B, offset:offset+B]
    int num_elements = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int r = i / BLOCK_SIZE;
        int c = i % BLOCK_SIZE;
        int global_r = col_offset + r;
        int global_c = col_offset + c;
        if (global_r < N && global_c < N) {
            sh_H[r * BLOCK_SIZE + c] = H[global_r * N + global_c];
        } else {
            sh_H[r * BLOCK_SIZE + c] = 0.0f;
        }
    }
    __syncthreads();

    // Compute inverse diag
    if (tid < BLOCK_SIZE) {
        float diag = sh_H[tid * BLOCK_SIZE + tid];
        sh_inv_diag[tid] = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
    }

    // Load weights to registers
    // Keep errors through to step 2
    float w_regs[BLOCK_SIZE];
    float err_regs[BLOCK_SIZE];

    if (row < M) {
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            if (col_offset + j < N) {
                w_regs[j] = W[row * N + (col_offset + j)];
            } else {
                w_regs[j] = 0.0f;
            }
        }
    }

    // quantize weights
    if (row < M) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            if (col_offset + j >= N) break;

            float w = w_regs[j];
            float s = Scales[col_offset + j];
            float z = Zeros[col_offset + j];

            float d = quantize_val(w, s, z, qmin, qmax);
            float err = w - q;

            w_regs[j] = q;
            err_regs[j] = err;

            float d_inv = sh_inv_diag[j];

            #pragma unroll
            for (int k = j + 1; k < BLOCK_SIZE; ++k) {
                float corr = sh_H[j * BLOCK_SIZE + k] * d_inv;
                w_regs[k] -= err * corr;
            }
        }

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            if (col_offset + j < N) {
                W[row * N + (col_offset + j)] = w_regs[j];
            }
        }
    }

    __syncthreads();

    // Step 2: propogate error to columns outside block

    int rem_start = col_offset + BLOCK_SIZE;
    
    for (int k_start = rem_start; k_start < N; k_start += TILE_WIDTH) {
        // Load H_tile to shared memory
        // H[col_offset : col_offset + B, k_start: k_start + TILE_WIDTH]

        int tile_elements = BLOCK_SIZE * TILE_WIDTH;
        for (int i = tid; i < tile_elements; i += blockDim.x) {
            int r = i / TILE_WIDTH;
            int c = i % TILE_WIDTH;

            int global_h_row = col_offset + r;
            int global_h_col = k_start + c;

            if (global_h_col < N) {
                float h_val = H[global_h_row * N + global_h_col];

                float scale = sh_inv_diag[r];
                sh_H[r * TILE_WIDTH + c] = h_val * scale;
            } else {
                sh_H[r * TILE_WIDTH + c] = 0.0f;
            }
        }
        __syncthreads();

        // Compute and update W
        if (row < M) {
            for (int k = 0; k < TILE_WIDTH; ++k) {
                int global_col = k_start + k;
                if (global_col >= N) break;

                float w_val = W[row * N + global_col];

                float correction = 0.0f;

                #pragma unroll
                for (int b = 0; b < BLOCK_SIZE; ++b) {
                    correction += err_regs[b] * sh_H[b * TILE_WIDTH + k];
                }
                
                W[row * N + global_col] = w_val - correction;
            }
        }
        __syncthreads();
    }
}


void gptq_fused_cuda(
        torch::Tensor W,
        torch::Tensor Err,
        torch::Tensor H,
        torch::Tensor Scales,
        torch::Tensor Zeros,
        int col_offset,
        float qmin,
        float qmax
        ) {
    int M = W.size(0);
    itn N = W.size(1);

    dim3 blockDim(256);
    dim3 gridDim((M + 255) / 256);

    int shared_mem_bytes = (BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE) * sizeof(float);

    gptq_fused_kernel<<<gridDim, blockDim, shared_mem_bytes>>>(
            W.data_ptr<float>(),
            Err.data_ptr<float>(),
            H.data_ptr<float>(),
            Scales.data_ptr<float>(),
            Zeros.data_ptr<float>(),
            M, N, col_offset, qmin, qmax
            );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
