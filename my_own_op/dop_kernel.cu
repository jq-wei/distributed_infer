#include <torch/extension.h>

template <typename scalar_t>
__global__ void dop_fw_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> mat_1,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> mat_2,
    float mul,  // scalar float
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> res
) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;


    if (row < mat_1.size(0) && col < mat_1.size(1)) {
        // Perform some operation on the matrix with `mul`
        res[row][col] = mat_1[row][col]* mul + mat_2[row][col];
    }
}


torch::Tensor dop_fw_cu(
    torch::Tensor mat_1,
    torch::Tensor mat_2,
    const float mul
){

    const int N = mat_1.size(0);
    const int M = mat_1.size(1);


    torch::Tensor res = torch::zeros({N,M}, mat_1.options());
    
    const dim3 threads(16,16); //256, row column prallel.
    const dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(mat_1.type(), "dop_fw_cu", 
    ([&] {
        dop_fw_kernel<scalar_t><<<blocks, threads>>>(
            mat_1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            mat_2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            mul,
            res.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));


    return res;
}