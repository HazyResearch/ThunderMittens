// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_fwd/attn_fwd.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
array attn_fwd(
    const array& q, // Input array q
    const array& k, // Input array k
    const array& v, // Input array q
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && k.dtype() == bfloat16);
    assert(q.shape() == k.shape() && k.shape() == v.shape());

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int N = q.shape(2);
    const int D = q.shape(3);
    
    
    assert(D == 64 || D == 128);

  // Promote dtypes between x and y as needed

  // Upcast to float32 for non-floating point inputs x and y
  auto out_dtype = bfloat16;

  // Broadcast the shapes of x and y (on the same stream s)
  auto out_shape = q.shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return array(
      /* const std::vector<int>& shape = */ out_shape,
      /* Dtype dtype = */ out_dtype,
      /* std::unique_ptr<Primitive> primitive = */
      std::make_shared<AttnFwd>(to_stream(s)),
      /* const std::vector<array>& inputs = */ {q, k, v});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void attn_fwd_impl(
    const array& q, // Input array q
    const array& k, // Input array k
    const array& v, // Input array q
    array& out) {
    assert(false); // no backup!
}

/** Fall back implementation for evaluation on CPU */
void AttnFwd::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {

  // Check the inputs (registered in the op while constructing the out array)
  assert(false);

//   assert(inputs.size() == 3);
//   auto& q = inputs[0];
//   auto& k = inputs[1];
//   auto& out = outputs[0];

//   // Dispatch to the correct dtype
//   if (out.dtype() == float32) {
//     return attn_fwd_impl<float>(x, y, out);
//   } else if (out.dtype() == float16) {
//     return attn_fwd_impl<float16_t>(x, y, out);
//   } else if (out.dtype() == bfloat16) {
//     return attn_fwd_impl<bfloat16_t>(x, y, out);
//   } else {
//     throw std::runtime_error(
//         "Axpby is only supported for floating point types.");
//   }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Accelerate Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on CPU falling back to common backend */
void AttnFwd::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}


///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

// #ifdef _METAL_

/** Evaluate primitive on GPU */
void AttnFwd::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Prepare inputs
  assert(inputs.size() == 3);
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& out = outputs[0];
    
  // Each primitive carries the stream it should execute on
  // and each stream carries its device identifiers
  auto& s = stream();
  // We get the needed metal device using the stream
  auto& d = metal::device(s.device);

//   // Prepare to specialize based on contiguity
//   bool contiguous_kernel =
//       (x.flags().row_contiguous && y.flags().row_contiguous) ||
//       (x.flags().col_contiguous && y.flags().col_contiguous);

//   if (contiguous_kernel) {
//     out.set_data(
//         allocator::malloc_or_wait(x.data_size() * out.itemsize()),
//         x.data_size(),
//         x.strides(),
//         x.flags());
//   } else {
//     out.set_data(allocator::malloc_or_wait(out.nbytes()));
//   }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  // Resolve name of kernel (corresponds to axpby.metal)
  std::ostringstream kname;
  const int B = q.shape(0);
  const int H = q.shape(1);
  const int N = q.shape(2);
  const int D = q.shape(3);
  kname << "attn_fwd_";
  kname << std::to_string(D);

  // Make sure the metal library is available
  d.register_library("mlx_ext");

  // Make a kernel from this metal library
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");

  // Prepare to encode kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Kernel parameters are registered with buffer indices corresponding to
  // those in the kernel declaration at axpby.metal
  int ndim = out.ndim();
  size_t nelem = out.size();

  // Encode input arrays to kernel
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
//  compute_encoder.set_bytes(H, 4);
//  compute_encoder.set_bytes(N, 5);
    compute_encoder.set_bytes(N, 4);
    compute_encoder.set_bytes(H, 5);


  MTL::Size group_dims = MTL::Size(32, 1, 1);

  
  MTL::Size grid_dims = MTL::Size(N / 8, H, B);

  // Launch the grid with the given number of threads divided among
  // the given threadgroups
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// #else // Metal is not available

// /** Fail evaluation on GPU */
// void Axpby::eval_gpu(
//     const std::vector<array>& inputs,
//     std::vector<array>& out) {
//   throw std::runtime_error("Axpby has no GPU implementation.");
// }

// #endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<array> AttnFwd::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
//   // Forward mode diff that pushes along the tangents
//   // The jvp transform on the primitive can built with ops
//   // that are scheduled on the same stream as the primitive

//   // If argnums = {0}, we only push along x in which case the
//   // jvp is just the tangent scaled by alpha
//   // Similarly, if argnums = {1}, the jvp is just the tangent
//   // scaled by beta
//   if (argnums.size() > 1) {
//     auto scale = argnums[0] == 0 ? alpha_ : beta_;
//     auto scale_arr = array(scale, tangents[0].dtype());
//     return {multiply(scale_arr, tangents[0], stream())};
//   }
//   // If, argnums = {0, 1}, we take contributions from both
//   // which gives us jvp = tangent_x * alpha + tangent_y * beta
//   else {
//     return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
//   }
    throw std::runtime_error("AttnFwd has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<array> AttnFwd::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
//   // Reverse mode diff
//   std::vector<array> vjps;
//   for (auto arg : argnums) {
//     auto scale = arg == 0 ? alpha_ : beta_;
//     auto scale_arr = array(scale, cotangents[0].dtype());
//     vjps.push_back(multiply(scale_arr, cotangents[0], stream()));
//   }
//   return vjps;
throw std::runtime_error("AddCustom has no vjp implementation.");
}

/** Vectorize primitive along given axis */
std::pair<std::vector<array>, std::vector<int>> AttnFwd::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("AddCustom has no vmap implementation.");
}

/** Equivalence check **/
bool AttnFwd::is_equivalent(const Primitive& other) const {
  const AttnFwd& r_other = static_cast<const AttnFwd&>(other);
  return true;
}

} // namespace mlx::core

