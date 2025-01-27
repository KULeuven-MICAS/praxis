builtin.module {
  func.func @snax_main() -> (tensor<16x16xi32>, tensor<16x16xi32>) {
    %0 = arith.constant dense<[[47, 89, -128, -71, 76, -65, -108, -93, 0, 72, -91, -110, 40, 62, 109, -84], [-78, -48, -127, 55, 57, -113, 102, 19, 113, 45, 5, -43, -82, -120, 107, -25], [-12, 123, 33, 51, 49, -112, 63, -70, -65, 38, 101, -72, 93, -91, 15, 69], [45, 82, 119, -111, -103, -10, -108, 51, 18, -3, -5, -36, 119, -104, -45, 115], [116, -48, -61, 120, 20, -18, 6, 93, 88, -78, 70, 51, 69, 115, 89, -75], [-33, 40, -62, 69, -7, 81, 50, 121, -9, 76, 37, 74, 107, 69, -63, 27], [98, -51, -84, -79, -125, -39, 48, -111, 60, 17, -114, 96, -36, 0, 46, -84], [-113, 107, -37, -24, -112, 83, 77, -37, 24, 50, -56, 23, 74, 100, -8, -17], [60, 49, 14, -120, 90, 4, -59, 11, 3, -29, -97, 37, -127, 100, -37, 98], [-94, -114, 14, -66, 40, 119, -19, -68, 96, 48, -78, 124, 81, -41, -107, 80], [5, -64, -94, 105, -46, -75, 76, -123, 92, 115, -114, 32, -19, 87, 90, -35], [-40, 74, 54, -120, -127, -21, 101, -127, 64, -14, -53, -21, -20, 89, 0, 51], [49, 8, -73, 75, -82, 62, 40, -20, 44, 38, 81, -50, -125, -108, 26, 115], [-72, -11, -47, -49, -102, 6, -105, -14, 100, 118, 41, -70, 93, 123, -46, 14], [-56, -27, 17, 59, 97, 106, -63, -95, -30, -84, 126, -47, -118, 55, -48, 21], [-70, -112, 19, -29, 125, -6, -101, -18, -95, -114, 38, 92, -122, 0, 53, 115]]> : tensor<16x16xi8>
    %1 = arith.constant dense<[[-88, 115, -78, 78, 122, 67, -70, -9, 126, -118, 0, 43, 105, 110, -105, 86], [-13, 21, -38, 19, -28, 37, 123, -69, -15, 74, -105, -34, 5, -46, -79, -101], [-41, 57, 77, -18, 88, -81, -95, 77, 1, -42, 15, -118, 81, 71, -55, -103], [-66, 47, -2, 47, -29, -102, -20, -54, 56, -71, -100, 106, -114, 98, -59, 125], [-46, 83, -100, 63, -81, 96, -90, 44, 31, 111, -61, -107, 15, -65, 55, -5], [-65, 21, 65, -69, -8, -109, -21, 79, -64, -38, -40, -86, 50, -33, -90, -7], [-24, 85, -94, 84, -64, -119, -46, 43, 16, 39, -127, -88, -93, -95, -88, 115], [-45, 92, 23, -67, 66, 89, 88, 42, 67, -110, 3, 67, -122, -36, 67, -101], [-57, -15, 56, -83, 108, 68, 2, -22, -78, 42, -84, 40, -58, -77, 52, -4], [-21, 96, 120, -123, 95, -37, -10, -24, -25, 77, 66, -9, -45, 95, -78, -52], [-38, 71, -106, 67, -75, -38, -84, 86, -107, -108, 3, 83, 107, -93, -113, 121], [85, 80, -60, 44, -22, 106, 65, -37, -89, -68, 92, -20, 38, -49, 80, 18], [-108, 24, 40, 95, -70, -36, -68, 11, 79, 28, 96, 59, 56, 3, -20, -16], [-120, 90, -43, 10, -24, 57, 38, 4, 38, -69, 122, 12, 59, -27, -58, -77], [72, 13, 76, 12, -108, 2, -73, 20, 30, 36, 63, -50, 82, -80, 57, 55], [-16, -16, 44, 69, 66, -62, -41, -81, 105, 64, -82, -116, -106, -1, 24, 25]]> : tensor<16x16xi8>
    %2 = arith.constant dense<[[2175, -15200, 7834, -9105, -16344, 39662, 7443, -18739, 16733, 46623, 26754, 8701, 28969, 6346, 8234, -21844], [30614, -13293, -1487, -9344, -17466, 4164, -4465, -8732, -11351, 38421, -38403, 11672, -51512, -28674, 33784, 41264], [-4797, 5278, -14668, 41187, -26001, -31022, -26682, -8277, 15842, 33386, -29480, -6509, -2084, 1626, -28328, 21653], [-6179, -17506, 37944, -5963, 45028, -3398, 1459, -7474, 16063, 2527, 6261, -6124, 11041, 21406, -341, -42195], [-34429, 39448, -27389, 24471, -15504, 30657, -9496, 7856, 17767, -51227, 15915, 50362, 14546, -15365, -1637, 33583], [-32219, 36761, -1843, 1704, -6613, -6134, 18779, -816, 2827, -16515, 4368, 17047, -25712, -10117, -18066, -6792], [30084, -14232, 968, -8390, 14863, 20756, 14820, -18500, -11178, 373, 25653, 8739, 15355, 7457, 11676, 17801], [-7403, -5149, 19657, -15200, -15106, -22000, 32401, -9497, -18811, 19135, 7658, -7744, -9338, -18611, -18650, -25950], [2703, 3785, -11295, -6880, 28824, 46150, 20363, -12918, 12970, 11142, -471, -41724, 4578, -3190, 19980, -39433], [1689, -14427, 29186, -17704, 17606, -7407, -6872, -1611, -26399, 27496, 9540, -31401, -11668, -4668, 27167, -14722], [6063, 1789, 14261, -7578, -1652, -3643, 156, -29382, 3854, 22725, 10389, 13082, -22215, 13087, 3677, 29733], [6301, -20722, 10025, -3441, 9538, -18012, 11227, -11194, -12247, 26383, -3216, -32651, 6544, -14619, -14837, -17829], [9167, -8260, 6231, -5738, 18932, -34854, -7096, -12160, -6752, -3473, -50566, 2811, -29613, 2878, -21960, 48266], [-24934, -12427, 37816, -34672, 19724, 536, 13944, -8138, -14658, 6005, 37584, 31710, 2233, 5109, -9460, -32532], [-7834, -8353, -24593, 6865, -23360, -18063, -18716, 17945, -25806, -11184, -21585, -12942, 20531, -11432, -17497, 19084], [35962, -14214, -16913, 17200, -23250, 19090, -14713, 6407, -5275, -3899, 11564, -34343, 13625, -18003, 47200, 9634]]> : tensor<16x16xi32>
    %3 = arith.constant 0 : i32
    %4 = tensor.empty() : tensor<16x16xi32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1, %3, %3 : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32) outs(%4 : tensor<16x16xi32>) {
    ^0(%arg0 : i8, %arg1 : i8, %arg2 : i32, %arg3 : i32, %arg4 : i32):
      %6 = arith.extsi %arg0 : i8 to i32
      %7 = arith.subi %6, %arg2 : i32
      %8 = arith.extsi %arg1 : i8 to i32
      %9 = arith.subi %8, %arg3 : i32
      %10 = arith.muli %7, %9 : i32
      %11 = arith.addi %arg4, %10 : i32
      linalg.yield %11 : i32
    } -> tensor<16x16xi32>
    func.return %5, %2 : tensor<16x16xi32>, tensor<16x16xi32>
  }
}
