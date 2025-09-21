/*
 * PyTorch C++ Extension Wrapper for Custom CUDA Kernels
 * Allows seamless integration with PyTorch models
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations of CUDA functions
extern "C" {
    void flash_attention_forward_cuda(
        const float* Q, const float* K, const float* V, float* output,
        int batch_size, int num_heads, int seq_len, int head_dim,
        cudaStream_t stream
    );
    
    void flash_attention_forward_fp16_cuda(
        const at::Half* Q, const at::Half* K, const at::Half* V, at::Half* output,
        int batch_size, int num_heads, int seq_len, int head_dim,
        cudaStream_t stream
    );
    
    void fused_matmul_gelu_cuda_fp16(
        const at::Half* A, const at::Half* B, const at::Half* bias, at::Half* C,
        int M, int N, int K, cudaStream_t stream
    );
    
    void fused_matmul_silu_cuda_fp16(
        const at::Half* A, const at::Half* B, const at::Half* bias, at::Half* C,
        int M, int N, int K, cudaStream_t stream
    );
    
    void int8_matmul_cuda(
        const int8_t* A, const int8_t* B,
        const float* scale_A, const float* scale_B, const float* bias, float* C,
        int M, int N, int K, cudaStream_t stream
    );
    
    void int4_matmul_cuda(
        const uint8_t* A, const uint8_t* B,
        const float* scale_A, const float* scale_B, const float* bias, float* C,
        int M, int N, int K, cudaStream_t stream
    );
}

// PyTorch wrapper for Flash Attention
torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, heads, seq_len, head_dim]");
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto output = torch::empty_like(Q);
    
    if (Q.dtype() == torch::kFloat16) {
        flash_attention_forward_fp16_cuda(
            reinterpret_cast<const at::Half*>(Q.data_ptr()),
            reinterpret_cast<const at::Half*>(K.data_ptr()),
            reinterpret_cast<const at::Half*>(V.data_ptr()),
            reinterpret_cast<at::Half*>(output.data_ptr()),
            batch_size, num_heads, seq_len, head_dim,
            at::cuda::getCurrentCUDAStream()
        );
    } else {
        flash_attention_forward_cuda(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            at::cuda::getCurrentCUDAStream()
        );
    }
    
    return output;
}

// PyTorch wrapper for Fused MatMul + GELU
torch::Tensor fused_linear_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Only FP16 supported for fused ops");
    
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    const at::Half* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
        bias_ptr = reinterpret_cast<const at::Half*>(bias.value().data_ptr());
    }
    
    fused_matmul_gelu_cuda_fp16(
        reinterpret_cast<const at::Half*>(input.data_ptr()),
        reinterpret_cast<const at::Half*>(weight.data_ptr()),
        bias_ptr,
        reinterpret_cast<at::Half*>(output.data_ptr()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

// PyTorch wrapper for Fused MatMul + SiLU
torch::Tensor fused_linear_silu(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Only FP16 supported for fused ops");
    
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    const at::Half* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = reinterpret_cast<const at::Half*>(bias.value().data_ptr());
    }
    
    fused_matmul_silu_cuda_fp16(
        reinterpret_cast<const at::Half*>(input.data_ptr()),
        reinterpret_cast<const at::Half*>(weight.data_ptr()),
        bias_ptr,
        reinterpret_cast<at::Half*>(output.data_ptr()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

// PyTorch wrapper for INT8 MatMul
torch::Tensor int8_linear(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be INT8");
    
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);
    
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    
    int8_matmul_cuda(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        scale_input.data_ptr<float>(),
        scale_weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

// PyTorch wrapper for INT4 MatMul
torch::Tensor int4_linear(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be packed UINT8");
    
    const int M = input.size(0);
    const int K = input.size(1) * 2;  // Packed INT4
    const int N = weight.size(1);
    
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    
    int4_matmul_cuda(
        input.data_ptr<uint8_t>(),
        weight.data_ptr<uint8_t>(),
        scale_input.data_ptr<float>(),
        scale_weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward, 
          "Flash Attention Forward (CUDA)");
    m.def("fused_linear_gelu", &fused_linear_gelu,
          "Fused Linear + GELU (CUDA)");
    m.def("fused_linear_silu", &fused_linear_silu,
          "Fused Linear + SiLU (CUDA)");
    m.def("int8_linear", &int8_linear,
          "INT8 Linear Layer (CUDA)");
    m.def("int4_linear", &int4_linear,
          "INT4 Linear Layer (CUDA)");
}
