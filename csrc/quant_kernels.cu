#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include "quant_kerenls.h"

__global__ void quant_and_pack_vcache_dim4_kernel(
    const half* __restrict__ value,
    int32_t* __restrict__ value_cache,
    half* __restrict__ value_scale,
    half* __restrict__ value_mn,
    const int num_blocks, const int num_heads, const int head_size, const int block_size,
    const int group_size, const int num_groups, const int bits, const int feat_per_int, const int max_int) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_blocks * num_heads * head_size) {
        const int block_idx = idx / (num_heads * head_size);
        const int head_idx = (idx / head_size) % num_heads;
        const int feat_idx = idx % head_size;

        for (int g = 0; g < num_groups; ++g) {
            int group_start = g * group_size;
            int group_end = group_start + group_size;

            half min_val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + feat_idx * block_size + group_start]; 
            half max_val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + feat_idx * block_size + group_start];

            for (int i = group_start; i < group_end; ++i) {
                half val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + feat_idx * block_size + i];
                min_val = __float2half(fminf(__half2float(min_val), __half2float(val)));
                max_val = __float2half(fmaxf(__half2float(max_val), __half2float(val)));
            }

            float span = __half2float(max_val) - __half2float(min_val);
            half sc = __float2half(span / max_int);
            value_scale[block_idx * num_heads * head_size * num_groups + head_idx * head_size * num_groups + feat_idx * num_groups + g] = sc;
            value_mn[block_idx * num_heads * head_size * num_groups + head_idx * head_size * num_groups + feat_idx * num_groups + g] = min_val;

            for (int i = group_start; i < group_end; ++i) {
                int quantized_val = static_cast<int>((__half2float(value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + feat_idx * block_size + i]) - __half2float(min_val)) / __half2float(sc));
                quantized_val = min(max(quantized_val, 0), max_int);
                int bit_offset = (i % feat_per_int) * bits;
                atomicOr(&value_cache[block_idx * num_heads * head_size * block_size / feat_per_int + head_idx * head_size * block_size / feat_per_int + feat_idx * block_size / feat_per_int + i / feat_per_int], quantized_val << bit_offset);
            }
        }
    }
}

__global__ void quant_and_pack_vcache_dim3_kernel(
    const half* __restrict__ value,
    int32_t* __restrict__ value_cache,
    half* __restrict__ value_scale,
    half* __restrict__ value_mn,
    const int num_blocks, const int num_heads, const int head_size, const int block_size,
    const int group_size, const int num_groups, const int bits, const int feat_per_int, const int max_int, const float threshold) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_blocks * num_heads * block_size) {
        const int block_idx = idx / (num_heads * block_size);
        const int head_idx = (idx / block_size) % num_heads;
        const int feat_idx = idx % block_size;

        for (int g = 0; g < num_groups; ++g) {
            int group_start = g * group_size;
            int group_end = group_start + group_size;

            half min_val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + group_start * block_size + feat_idx];
            half max_val = min_val;

            for (int i = group_start; i < group_end; ++i) {
                half val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + i * block_size + feat_idx];
                min_val = __float2half(fminf(__half2float(min_val), __half2float(val)));
                max_val = __float2half(fmaxf(__half2float(max_val), __half2float(val)));
            }

            float span = __half2float(max_val) - __half2float(min_val);
            half sc = __float2half(span / max_int);
            int scale_index = block_idx * num_heads * block_size * num_groups + head_idx * block_size * num_groups + g * block_size + feat_idx;

            if (fabsf(__half2float(sc) - __half2float(value_scale[scale_index])) > threshold ||
                fabsf(__half2float(min_val) - __half2float(value_mn[scale_index])) > threshold) {

                value_scale[scale_index] = sc;
                value_mn[scale_index] = min_val;

                for (int i = group_start; i < group_end; ++i) {
                    half val = value[block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + i * block_size + feat_idx];
                    int quantized_val = static_cast<int>((__half2float(val) - __half2float(min_val)) / __half2float(sc));
                    quantized_val = min(max(quantized_val, 0), max_int);
                    int bit_offset = (i % feat_per_int) * bits;
                    atomicOr(&value_cache[block_idx * num_heads * head_size * block_size / feat_per_int + head_idx * head_size * block_size / feat_per_int + (i / feat_per_int) * block_size + feat_idx], quantized_val << bit_offset);
                }
            }
        }
    }
}

void quant_and_pack_dim3_cuda(
    torch::Tensor v, torch::Tensor scale, torch::Tensor mn, torch::Tensor code,
    const int group_size, const int bits) {

    const auto shape = v.sizes();
    const int num_blocks = shape[0];
    const int num_heads = shape[1];
    const int head_size = shape[2];
    const int block_size = shape[3];
    const int num_groups = head_size / group_size;
    const int feat_per_int = 32 / bits;
    const int max_int = (1 << bits) - 1;

    const int threadsPerBlock = 256;
    const int total_elements = num_blocks * num_heads * block_size;
    const int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    quant_and_pack_vcache_dim3_kernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<half*>(v.data_ptr()),
        reinterpret_cast<int32_t*>(code.data_ptr()),
        reinterpret_cast<half*>(scale.data_ptr()),
        reinterpret_cast<half*>(mn.data_ptr()),
        num_blocks, num_heads, head_size, block_size, group_size, num_groups, bits, feat_per_int, max_int, 0.03);
}

void quant_and_pack_vcache_cuda(
    torch::Tensor v, torch::Tensor scale, torch::Tensor mn, torch::Tensor code,
    const int group_size, const int bits) {

    const auto shape = v.sizes();
    const int num_blocks = shape[0];
    const int num_heads = shape[1];
    const int head_size = shape[2];
    const int block_size = shape[3];
    const int num_groups = head_size / group_size;
    const int feat_per_int = 32 / bits;
    const int max_int = (1 << bits) - 1;

    const int threadsPerBlock = 256;
    const int total_elements = num_blocks * num_heads * head_size;
    const int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    quant_and_pack_vcache_dim4_kernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<half*>(v.data_ptr()),
        reinterpret_cast<int32_t*>(code.data_ptr()),
        reinterpret_cast<half*>(scale.data_ptr()),
        reinterpret_cast<half*>(mn.data_ptr()),
        num_blocks, num_heads, head_size, block_size, group_size, num_groups, bits, feat_per_int, max_int);
}

__global__ void unpack_and_dequant_vcache_kernel(const int32_t* __restrict__ v_code,
                                            const half* __restrict__ scale,
                                            const half* __restrict__ mn,
                                            half* __restrict__ data,
                                            int num_elements,
                                            int group_size,
                                            int bits,
                                            int feat_per_int) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {

        int packed_idx = idx / feat_per_int;
        int bit_offset = (idx % feat_per_int) * bits;
        int num = 0xFF >> (8 - bits);

        int v_code_val = reinterpret_cast<const int*>(v_code)[packed_idx];
        int unpacked_val = (v_code_val >> bit_offset) & num;

        int group_idx = idx / group_size;
        half scale_val = scale[group_idx];
        half mn_val = mn[group_idx];

        float unpacked_val_float = static_cast<float>(unpacked_val);

        data[idx] = __float2half(unpacked_val_float * __half2float(scale_val) + __half2float(mn_val));
    }
}

__global__ void unpack_and_dequant_vcache_dim3_kernel(const int32_t* __restrict__ v_code,
                                            const half* __restrict__ scale,
                                            const half* __restrict__ mn,
                                            half* __restrict__ data,
                                            int num_elements,
                                            int group_size,
                                            int bits,
                                            int feat_per_int,
                                            int last_dim) {
                
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        
        int reminder = idx % last_dim;
        int dim3idx = idx / last_dim;
        int packed_idx = dim3idx / feat_per_int;
        int bit_offset = (dim3idx % feat_per_int) * bits;
        int num = 0xFF >> (8 - bits);

        int v_code_val = reinterpret_cast<const int*>(v_code)[packed_idx * last_dim + reminder];
        int unpacked_val = (v_code_val >> bit_offset) & num;

        int group_idx = dim3idx / group_size;
        half scale_val = scale[group_idx * last_dim + reminder];
        half mn_val = mn[group_idx * last_dim + reminder];

        float unpacked_val_float = static_cast<float>(unpacked_val);
        data[dim3idx * last_dim + reminder] = __float2half(unpacked_val_float * __half2float(scale_val) + __half2float(mn_val));
    }
}

void unpack_and_dequant_vcache_dim3_cuda(
    torch::Tensor v_code, torch::Tensor scale, torch::Tensor mn, torch::Tensor data,
    const int group_size, const int bits) {
    
    const auto shape = data.sizes();
    const int W = shape[3];
    const int feat_per_int = 32 / bits;

    int num_elements = data.numel();
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    unpack_and_dequant_vcache_dim3_kernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const int32_t*>(v_code.data_ptr()),
        reinterpret_cast<const half*>(scale.data_ptr()),
        reinterpret_cast<const half*>(mn.data_ptr()),
        reinterpret_cast<half*>(data.data_ptr()),
        num_elements, group_size, bits, feat_per_int, W);
}

void unpack_and_dequant_vcache_cuda(
    torch::Tensor v_code, torch::Tensor scale, torch::Tensor mn, torch::Tensor data,
    const int group_size, const int bits) {

    const int feat_per_int = 32 / bits;

    int num_elements = data.numel();
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    unpack_and_dequant_vcache_kernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const int32_t*>(v_code.data_ptr()),
        reinterpret_cast<const half*>(scale.data_ptr()),
        reinterpret_cast<const half*>(mn.data_ptr()),
        reinterpret_cast<half*>(data.data_ptr()),
        num_elements, group_size, bits, feat_per_int);
}