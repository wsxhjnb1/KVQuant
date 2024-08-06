#pragma once
#include <torch/extension.h>

void quant_and_pack_vcache_cuda(
    torch::Tensor v, torch::Tensor scale, torch::Tensor mn, torch::Tensor code,
    const int group_size, const int bits);

void quant_and_pack_dim3_cuda(
    torch::Tensor v, torch::Tensor scale, torch::Tensor mn, torch::Tensor code,
    const int group_size, const int bits);

void unpack_and_dequant_vcache_dim3_cuda(
    torch::Tensor v_code, torch::Tensor scale, torch::Tensor mn, torch::Tensor data,
    const int group_size, const int bits);

void unpack_and_dequant_vcache_cuda(
    torch::Tensor v_code, torch::Tensor scale, torch::Tensor mn, torch::Tensor data,
    const int group_size, const int bits);
