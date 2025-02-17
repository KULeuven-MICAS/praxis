import pickle
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

from praxis.backend.zigzag import (
    process_cme,
)


@dataclass
class ApplyZigzagScheduleRewriter(RewritePattern):
    zigzag_cme_path: str | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: GenericOp, rewriter: PatternRewriter):
        if self.zigzag_cme_path is None:
            raise ValueError("Path to zigzag CME pickle file is mandatory.")
        with open(self.zigzag_cme_path, "rb") as fp:
            cmes = pickle.load(fp)
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
class ApplyZigzagSchedule(ModulePass):
    name = "apply-zigzag-schedule"
    zz_cme: str | None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ApplyZigzagScheduleRewriter(zigzag_cme_path=self.zz_cme),
            apply_recursively=False,
        ).rewrite_module(op)
