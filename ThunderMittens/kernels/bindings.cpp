//// Copyright © 2023-2024 Apple Inc.
//
//#include <nanobind/nanobind.h>
//#include <nanobind/stl/variant.h>
//
//#include "add_rt/add_rt.h"
//#include "attn_fwd/attn_fwd.h"
//#include "matmul_custom/matmul_custom.h"
//
//namespace nb = nanobind;
//using namespace nb::literals;
//
//using namespace mlx::core;
//
//NB_MODULE(_ext, m) {
//  m.doc() = "TK extension for MLX";
//      m.def(
//      "add_rt",
//      &add_rt,
//      "x"_a,
//      "y"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        adds
//      )");
//
//    m.def(
//      "attn_fwd",
//      &attn_fwd,
//      "q"_a,
//      "k"_a,
//      "v"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        attn fwd
//      )");
//
//    m.def(
//      "matmul_custom",
//      &matmul_custom,
//      "x"_a,
//      "y"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        gemm
//      )");
//}

// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "add_rt/add_rt.h"
#include "attn_fwd/attn_fwd.h"
#include "matmul_custom/matmul_custom.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(_ext, m) {
  m.doc() = "TK extension for MLX";
      m.def(
      "add_rt",
      &add_rt,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        adds
      )");

    m.def(
      "attn_fwd",
      &attn_fwd,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        attn fwd
      )");

    m.def(
      "matmul_custom",
      &matmul_custom,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        gemm
      )");
}
