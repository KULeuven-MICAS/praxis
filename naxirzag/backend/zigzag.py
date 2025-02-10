import importlib.resources
from xdsl.dialects.transform import TileOp
import yaml
from typing import IO, Any, cast
from xdsl.dialects.builtin import IntegerType, ModuleOp, ShapedType, ContainerType
from xdsl.dialects.linalg import GenericOp
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from xdsl.parser import DenseArrayBase
from xdsl.ir import SSAValue

from zigzag.api import get_hardware_performance_zigzag
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.mapping.utils import get_temporal_loops, get_spatial_loops


def generate_zigzag_workload(generic_op: GenericOp):
    assert len(generic_op.outputs) == 1
    # extract output operand and relevant indexing maps
    output_operand = generic_op.outputs[0]
    output_map = generic_op.indexing_maps.data[-1].data

    # make some assertions on correct inputs of the linalg generic
    shaped_inputs = [
        (op, map.data)
        for (op, map) in zip(generic_op.inputs, generic_op.indexing_maps.data[:-1])
        if isinstance(op.type, ShapedType)
    ]

    # input a, input b, output
    operands = [shaped_inputs[0][0], shaped_inputs[1][0], output_operand]
    indexing_maps = [shaped_inputs[0][1], shaped_inputs[1][1], output_map]

    zigzag_description = dict()
    zigzag_description["id"] = 0

    # for now, set operator to default type
    zigzag_description["operator_type"] = "Gemm"

    # construct equation
    output_access = "O"
    for i in range(len(indexing_maps[-1].results)):
        map = indexing_maps[-1].results[i]
        assert isinstance(map, AffineDimExpr)
        output_access += f"[{str(map)}]"

    input_i_access = "I"
    for i in range(len(indexing_maps[0].results)):
        map = indexing_maps[0].results[i]
        assert isinstance(map, AffineDimExpr)
        input_i_access += f"[{str(map)}]"

    input_w_access = "W"
    for i in range(len(indexing_maps[1].results)):
        map = indexing_maps[1].results[i]
        assert isinstance(map, AffineDimExpr)
        input_w_access += f"[{str(map)}]"

    # assume MAC
    zigzag_description[
        "equation"
    ] = f"{output_access}+={input_i_access}*{input_w_access}"

    # extract dimension_relations
    # for matmul, this is empty
    zigzag_description["dimension_relations"] = []

    # extract loop bounds by evaluating the inverse affine map
    # with the operand shapes as input
    results: list[AffineExpr] = []
    results.extend(indexing_maps[0].results)
    results.extend(indexing_maps[1].results)
    results.extend(indexing_maps[2].results)

    combined_affine_map = AffineMap(3, 0, tuple(results))
    inverse_map = combined_affine_map.inverse_permutation()
    assert inverse_map is not None

    operand_shapes = []
    for op in operands:
        assert isinstance(operand_type := op.type, ShapedType)
        operand_shapes.extend(operand_type.get_shape())

    iteration_bounds = inverse_map.eval(operand_shapes, [])

    zigzag_description["loop_dims"] = [f"D{i}" for i in range(len(iteration_bounds))]
    zigzag_description["loop_sizes"] = [x for x in iteration_bounds]

    # extract operand precision
    widths = []
    for op in operands:
        assert isinstance(operand_type := op.type, ContainerType)
        element_type = operand_type.get_element_type()
        if isinstance(element_type, IntegerType):
            widths.append(element_type.width.data)
        else:
            widths.append(element_type.get_bitwidth)

    zigzag_description["operand_precision"] = dict()
    zigzag_description["operand_precision"]["O"] = widths[-1]
    zigzag_description["operand_precision"]["O_final"] = widths[-1]
    zigzag_description["operand_precision"]["W"] = widths[0]
    zigzag_description["operand_precision"]["I"] = widths[1]

    # operand source (use default of no source for now)
    zigzag_description["operand_source"] = dict()
    zigzag_description["operand_source"]["W"] = 0
    zigzag_description["operand_source"]["I"] = 0

    # affects last two indices of input I
    workload = [zigzag_description]

    return workload


def process_cme(cme: CostModelEvaluation, target: SSAValue):
    spatial_loops = get_spatial_loops(cme)
    temporal_loops = get_temporal_loops(cme)

    all_tiling_ops: list[TileOp] = []

    tile_sizes = {x: 1 for x in cme.layer.layer_dims}

    for dim, size, mem_layer in spatial_loops:
        tile_sizes[dim] = int(tile_sizes[dim] * size[1])

    for dim, size, mem_layer in temporal_loops:
        if "l3" in mem_layer:
            tile_op = TileOp(
                target=target,
                dynamic_sizes=[],
                scalable_sizes=DenseArrayBase.create_dense_int(
                    IntegerType(1), [0, 0, 0]
                ),
                static_sizes=DenseArrayBase.create_dense_int(
                    IntegerType(64), [tile_sizes[dim] for dim in cme.layer.layer_dims]
                ),
            )
            all_tiling_ops.append(tile_op)
        tile_sizes[dim] = int(tile_sizes[dim] * size[1])

    # link them together
    for i in range(len(all_tiling_ops) - 1):
        all_tiling_ops[i].operands[0] = all_tiling_ops[i + 1].tiled_linalg_op

    return all_tiling_ops


def naxirzag_zigzag_wrapper(
    workload_path: str = "workload.yaml",
    hardware_path: str | None = None,
    mapping_path: str | None = None,
):
    """
    Simple wrapper for setting defaults in zigzag api
    """
    if hardware_path is None:
        hardware_path = str(
            importlib.resources.files("zigzag.inputs.hardware") / "gemm_l1_l3.yaml"
        )
    if mapping_path is None:
        mapping_path = str(
            importlib.resources.files("zigzag.inputs.mapping") / "gemm_l1_l3.yaml"
        )

    returned_values = get_hardware_performance_zigzag(
        workload_path, hardware_path, mapping_path
    )
    # The function can return 5 or 3 values, we want 3
    assert len(returned_values) == 3
    energy_total, latency_total, cmes = returned_values
    cmes = cast(list[tuple[CostModelEvaluation, Any]], cmes[0][1])
    return (energy_total, latency_total, cmes)


def print_total_cycles(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        if isinstance(op, GenericOp):
            generic_op = op
            if len(generic_op.outputs) != 1:
                return
            if not isinstance(generic_op.outputs[0].type, ShapedType):
                return

            # generate zigzag workload
            workload = generate_zigzag_workload(generic_op)

            # run zigzag
            with open("workload.yaml", "w") as f:
                f.write(yaml.dump(workload, sort_keys=False))

            hardware_path = (
                importlib.resources.files("zigzag.inputs.hardware") / "gemm_l1_l3.yaml"
            )
            mapping_path = (
                importlib.resources.files("zigzag.inputs.mapping") / "gemm_l1_l3.yaml"
            )

            returned_values = get_hardware_performance_zigzag(
                "workload.yaml", str(hardware_path), str(mapping_path)
            )

            # The function can return 5 or 3 values, we want 3
            assert len(returned_values) == 3
            energy_total, latency_total, cmes = returned_values
            cmes = cast(list[tuple[CostModelEvaluation, Any]], cmes[0][1])
            print(int(latency_total), file=output)
