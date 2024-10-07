#include <torch/extension.h>
#include "utils.h"

torch::Tensor dop(
    torch::Tensor mat_1,
    torch::Tensor mat_2,
    const float mul
){
    CHECK_INPUT(mat_1);
    CHECK_INPUT(mat_2);

    return dop_fw_cu(mat_1, mat_2, mul);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("dop", &dop);
}