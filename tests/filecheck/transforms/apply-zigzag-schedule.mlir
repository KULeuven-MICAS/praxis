// RUN: naxirzag-opt -p preprocess -t zigzag -o test_cme.pickle %s && naxirzag-opt -p preprocess,apply-zigzag-schedule{zz_cme=\"test_cme.pickle\"} %s | filecheck %s
module {
  func.func @snax_main() -> (tensor<16x16xi32>, tensor<16x16xi32>) {
    %cst = arith.constant dense<0> : tensor<16x16xi8>
    %cst_1 = arith.constant dense<0> : tensor<16x16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<16x16xi32>
    %1 = linalg.quantized_matmul ins(%cst, %cst, %c0_i32, %c0_i32 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%0 : tensor<16x16xi32>) -> tensor<16x16xi32>
    return %1, %cst_1 : tensor<16x16xi32>, tensor<16x16xi32>
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @snax_main() -> (tensor<16x16xi32>, tensor<16x16xi32>) {
// CHECK-NEXT:     %0 = arith.constant dense<0> : tensor<16x16xi8>
// CHECK-NEXT:     %1 = arith.constant dense<0> : tensor<16x16xi32>
// CHECK-NEXT:     %2 = arith.constant 0 : i32
// CHECK-NEXT:     %3 = tensor.empty() : tensor<16x16xi32>
// CHECK-NEXT:     %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %0, %2, %2 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%3 : tensor<16x16xi32>) attrs =  {zigzag_id = 0 : index} {
// CHECK-NEXT:     ^0(%arg0 : i8, %arg1 : i8, %arg2 : i32, %arg3 : i32, %arg4 : i32):
// CHECK-NEXT:       %5 = arith.extsi %arg0 : i8 to i32
// CHECK-NEXT:       %6 = arith.subi %5, %arg2 : i32
// CHECK-NEXT:       %7 = arith.extsi %arg1 : i8 to i32
// CHECK-NEXT:       %8 = arith.subi %7, %arg3 : i32
// CHECK-NEXT:       %9 = arith.muli %6, %8 : i32
// CHECK-NEXT:       %10 = arith.addi %arg4, %9 : i32
// CHECK-NEXT:       linalg.yield %10 : i32
// CHECK-NEXT:     } -> tensor<16x16xi32>
// CHECK-NEXT:     func.return %4, %1 : tensor<16x16xi32>, tensor<16x16xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "transform.named_sequence"() <{sym_name = "__transform_main", function_type = (!transform.any_op, !transform.op<"linalg.generic">) -> ()}> ({
// CHECK-NEXT:   ^0(%0 : !transform.any_op, %1 : !transform.op<"linalg.generic">):
// CHECK-NEXT:     %2 = "transform.structured.match"(%0) <{op_attrs = {zigzag_id = 0 : index}}> : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:     "transform.yield"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
