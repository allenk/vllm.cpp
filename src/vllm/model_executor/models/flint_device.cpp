// Flint device-kernel resolver: the OpProvider lookup for the kFlint glue table
// (registered by cuda_flint.cu on kCUDA). Always compiled; holds no CUDA code.
#include "vllm/model_executor/models/flint_device.h"

#include "vt/ops.h"  // OpId, GetOp, OpRegistered

namespace vllm::flint {

const FlintDeviceKernels* FlintDevice() {
  return static_cast<const FlintDeviceKernels*>(
      vt::GetOp(vt::OpId::kFlint, vt::DeviceType::kCUDA));
}

bool FlintDeviceKernelsAvailable() {
  return vt::OpRegistered(vt::OpId::kFlint, vt::DeviceType::kCUDA);
}

}  // namespace vllm::flint
