from xdsl.dialects import builtin
from xdsl.dialects.linalg import GenericOp
from xdsl.context import MLContext
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    ShapedType,
)
from xdsl.dialects.transform import (
    NamedSequenceOp,
    MatchOp,
    YieldOp,
    AnyOpType,
    OperationType,
)
from dataclasses import dataclass

from xdsl.rewriter import InsertPoint
import yaml

from naxirzag.backend.zigzag import (
    generate_zigzag_workload,
    naxirzag_zigzag_wrapper,
    process_cme,
)


@dataclass
class LinalgToStreamTranslator(RewritePattern):
    zigzag_hardware_path: str | None
    zigzag_mapping_path: str | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: GenericOp, rewriter: PatternRewriter):
        if len(generic_op.outputs) != 1:
            return
        if not isinstance(generic_op.outputs[0].type, ShapedType):
            return

        # generate zigzag workload
        workload = generate_zigzag_workload(generic_op)
        workload_path = "workload.yaml"
        with open(workload_path, "w") as f:
            f.write(yaml.dump(workload, sort_keys=False))

        _, _, cmes = naxirzag_zigzag_wrapper(
            workload_path, self.zigzag_hardware_path, self.zigzag_mapping_path
        )
        # for now, the assumption is 1 layer, with the following id:
        id = 0

        # Specify the tiling sequence:
        transform_inputs = [
            AnyOpType(),
            OperationType("linalg.generic"),
        ]

        function_type = builtin.FunctionType.from_lists(transform_inputs, [])
        sequence_op = NamedSequenceOp(
            "__transform_main",
            function_type,
            Region(Block([YieldOp()], arg_types=transform_inputs)),
        )

        module_op = generic_op
        while not isinstance(module_op, ModuleOp):
            module_op = module_op.parent_op()
            assert module_op

        rewriter.insert_op(sequence_op, InsertPoint.at_end(module_op.body.block))

        structured_match = MatchOp(
            target=sequence_op.body.block.args[0],
            op_attrs={"zigzag_id": IntegerAttr.from_index_int_value(id)},
        )
        rewriter.insert_op(
            structured_match,
            insertion_point=InsertPoint.at_start(sequence_op.body.block),
        )

        # generate tiling ops based on the cme:
        all_tiling_ops = process_cme(cmes[0][0], structured_match.results[0])
        # add stream id attribute to the generic op
        generic_op.attributes["zigzag_id"] = IntegerAttr.from_index_int_value(id)

        rewriter.insert_op(
            list(reversed(all_tiling_ops)), InsertPoint.after(structured_match)
        )


@dataclass(frozen=True)
class LinalgToStream(ModulePass):
    name = "linalg-to-stream"
    zz_hw: str | None
    zz_map: str | None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LinalgToStreamTranslator(
                zigzag_hardware_path=self.zz_hw, zigzag_mapping_path=self.zz_map
            ),
            apply_recursively=False,
        ).rewrite_module(op)
