// Flint device-kernel resolver: the OpProvider lookup for the kFlint glue table
// (registered by cuda_flint.cu on kCUDA), asked by the caller's device. Always
// compiled; holds no CUDA code.
#include "vllm/model_executor/models/flint_device.h"

#include "vt/ops.h"  // OpId, GetOp, OpRegistered

namespace vllm::flint {

const FlintDeviceKernels* FlintDevice(vt::DeviceType device) {
  return static_cast<const FlintDeviceKernels*>(vt::GetOp(vt::OpId::kFlint, device));
}

bool FlintDeviceKernelsAvailable(vt::DeviceType device) {
  return vt::OpRegistered(vt::OpId::kFlint, device);
}

}  // namespace vllm::flint
