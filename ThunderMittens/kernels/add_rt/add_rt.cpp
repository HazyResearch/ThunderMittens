// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "add_rt/add_rt.h"

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
array add_rt(
    const array& x, // Input array x
    const array& y, // Input array y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  // Promote dtypes between x and y as needed
  auto promoted_dtype = promote_types(x.dtype(), y.dtype());
  assert(x.dtype() == bfloat16 && y.dtype() == bfloat16);

  // Upcast to float32 for non-floating point inputs x and y
  auto out_dtype = bfloat16;

  // Cast x and y up to the determined dtype (on the same stream s)
    auto x_casted = astype(x, out_dtype, s);
    auto y_casted = astype(y, out_dtype, s);


  // Broadcast the shapes of x and y (on the same stream s)
  auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
  auto out_shape = broadcasted_inputs[0].shape();

  // Construct the array as the output of the Axpby primitive
  // with the broadcasted and upcasted arrays as inputs
  return array(
      /* const std::vector<int>& shape = */ out_shape,
      /* Dtype dtype = */ out_dtype,
      /* std::unique_ptr<Primitive> primitive = */
      std::make_shared<AddRT>(to_stream(s)),
      /* const std::vector<array>& inputs = */ broadcasted_inputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void add_rt_impl(
    const array& x,
    const array& y,
    array& out) {
  // We only allocate memory when we are ready to fill the output
  // malloc_or_wait synchronously allocates available memory
  // There may be a wait executed here if the allocation is requested
  // under memory-pressured conditions
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Collect input and output data pointers
  const T* x_ptr = x.data<T>();
  const T* y_ptr = y.data<T>();
  T* out_ptr = out.data<T>();

  // Cast alpha and beta to the relevant types

  // Do the element-wise operation for each output
  for (size_t out_idx = 0; out_idx < out.size(); out_idx++) {
    // Map linear indices to offsets in x and y
    auto x_offset = elem_to_loc(out_idx, x.shape(), x.strides());
    auto y_offset = elem_to_loc(out_idx, y.shape(), y.strides());

    // We allocate the output to be contiguous and regularly strided
    // (defaults to row major) and hence it doesn't need additional mapping
    out_ptr[out_idx] =  x_ptr[x_offset] + y_ptr[y_offset];
  }
}

/** Fall back implementation for evaluation on CPU */
void AddRT::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {

  // Check the inputs (registered in the op while constructing the out array)
  assert(false);
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Dispatch to the correct dtype
  if (out.dtype() == float32) {
    return add_rt_impl<float>(x, y, out);
  } else if (out.dtype() == float16) {
    return add_rt_impl<float16_t>(x, y, out);
  } else if (out.dtype() == bfloat16) {
    return add_rt_impl<bfloat16_t>(x, y, out);
  } else {
    throw std::runtime_error(
        "Axpby is only supported for floating point types.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Accelerate Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on CPU falling back to common backend */
void AddRT::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}


///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

// #ifdef _METAL_

/** Evaluate primitive on GPU */
void AddRT::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Prepare inputs
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // Each primitive carries the stream it should execute on
  // and each stream carries its device identifiers
  auto& s = stream();
  // We get the needed metal device using the stream
  auto& d = metal::device(s.device);

  // Prepare to specialize based on contiguity
  bool contiguous_kernel =
      (x.flags().row_contiguous && y.flags().row_contiguous) ||
      (x.flags().col_contiguous && y.flags().col_contiguous);

  if (contiguous_kernel) {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * out.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }


  // Resolve name of kernel (corresponds to axpby.metal)
  std::ostringstream kname;
  kname << "add_rt_";
  kname << type_to_name(out);

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
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(y, 1);

  // Encode output arrays to kernel
  compute_encoder.set_output_array(out, 2);

  compute_encoder.set_bytes(out.shape()[0], 3);
  compute_encoder.set_bytes(out.shape()[1], 4);

  // We launch 1 thread for each input and make sure that the number of
  // threads in any given threadgroup is not higher than the max allowed
  size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());

  // Fix the 3D size of each threadgroup (in terms of threads)
  MTL::Size group_dims = MTL::Size(32, 1, 1);

  // Fix the 3D size of the launch grid (in terms of threads)
  MTL::Size grid_dims = MTL::Size(out.shape()[1] / 8, out.shape()[0] / 8, 1);

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
std::vector<array> AddRT::jvp(
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
    throw std::runtime_error("AddRT has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<array> AddRT::vjp(
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
std::pair<std::vector<array>, std::vector<int>> AddRT::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("AddCustom has no vmap implementation.");
}

/** Equivalence check **/
bool AddRT::is_equivalent(const Primitive& other) const {
  const AddRT& r_other = static_cast<const AddRT&>(other);
  return true;
}

} // namespace mlx::core
