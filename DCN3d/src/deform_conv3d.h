#pragma once

#include "cpu/deform_conv3d_cpu.h"

#ifdef WITH_CUDA
#include "cuda/deform_conv3d_cuda.h"
#endif


at::Tensor
deform_conv3d_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const int kernel_t,
               const int kernel_h,
               const int kernel_w,
               const int stride_t,
               const int stride_h,
               const int stride_w,
               const int pad_t,
               const int pad_h,
               const int pad_w,
               const int dilation_t,
               const int dilation_h,
               const int dilation_w,
               const int group, 
               const int deformable_group,
               const int im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_conv3d_cuda_forward(input, weight, bias, offset, mask,
                                   kernel_t, kernel_h, kernel_w,
                                   stride_t, stride_h, stride_w,
                                   pad_t, pad_h, pad_w,
                                   dilation_t, dilation_h, dilation_w,
                                   group,
                                   deformable_group,
                                   im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
deform_conv3d_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                const int kernel_t, 
                const int kernel_h, 
                const int kernel_w,
                const int stride_t,
                const int stride_h, 
                const int stride_w,
                const int pad_t, 
                const int pad_h, 
                const int pad_w,
                const int dilation_t, 
                const int dilation_h, 
                const int dilation_w,
                const int group,
                const int deformable_group,
                const int im2col_step)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_conv3d_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_t, kernel_h, kernel_w,
                                    stride_t, stride_h, stride_w,
                                    pad_t, pad_h, pad_w,
                                    dilation_t, dilation_h, dilation_w,
                                    group,
                                    deformable_group,
                                    im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

