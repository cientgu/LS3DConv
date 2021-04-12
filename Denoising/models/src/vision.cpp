#include "deform_conv3d.h"
#include "modulated_deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
  m.def("deform_conv3d_forward", &deform_conv3d_forward, "deform_conv3d_forward");
  m.def("deform_conv3d_backward", &deform_conv3d_backward, "deform_conv3d_backward");
}
