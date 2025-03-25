import argparse
import sys
from typing import IO
from io import StringIO, BytesIO
from xdsl.dialects.builtin import ModuleOp
from snaxc.tools.snax_opt_main import SNAXOptMain

from praxis.transforms import get_all_passes
from praxis.backend.zigzag import dump_zigzag_workload, get_zigzag_cme


class PraxisMain(SNAXOptMain):
    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        """
        Registers all the command line arguments that are used by this tool.
        Add other/additional arguments by overloading this function.
        """
        super().register_all_arguments(arg_parser)

        arg_parser.add_argument(
            "--zigzag-hw",
            type=str,
            required=False,
            help="path to zigzag hardware file" "containing accelerator description",
        )

        arg_parser.add_argument(
            "--zigzag-map",
            type=str,
            required=False,
            help="path to zigzag mapping file, containing mapping constraints",
        )

        arg_parser.add_argument(
            "--zigzag-verbose",
            action="store_true",
            required=False,
            help="Run zigzag in verbose mode",
        )

    def register_all_passes(self):
        # Register all snax-opt passes (which includes all xdsl passes)
        super().register_all_passes()
        # Aditionally, register all praxis passes
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)

    def prepare_output(self) -> IO:
        # Override zigzag to always write to a binary file
        if self.args.target == "zigzag":
            if self.args.output_file is None:
                raise ValueError(
                    "Zigzag target requires an output file (-o) to be specified"
                )
            else:
                return open(self.args.output_file, "wb")
        if self.args.output_file is None:
            return sys.stdout
        else:
            return open(self.args.output_file, "w")

    # TODO, update output_resulting_program upstream to allow for bytes as output
    def output_resulting_program(  # pyright: ignore
        self, prog: ModuleOp
    ) -> str | bytes:
        """Get the resulting program.
        This version is adapted to, upon -t zigzag:
        - write to a binary stream
        - call zigzag with cli arguments (not typically done for targets)"
        And to, upon -t zigzag-workload
        - write to a yaml string stream
        - call the backend function to convert workloads to a yaml file"
        """
        if self.args.target not in self.available_targets:
            raise Exception(f"Unknown target {self.args.target}")
        if self.args.target == "zigzag":
            # Zigzag writes to a pickle file, which is binary
            output = BytesIO()
            get_zigzag_cme(
                prog,
                output,
                self.args.zigzag_hw,
                self.args.zigzag_map,
                self.args.zigzag_verbose,
            )
        else:
            output = StringIO()
            if self.args.target == "zigzag-workload":
                dump_zigzag_workload(
                    prog,
                    output,
                )
            else:
                self.available_targets[self.args.target](prog, output)
        return output.getvalue()

    def register_all_targets(self):
        """Register all regular targets, and an empty zigzag target,
        The target is empty, because the inherited output_resulting_program
        function can not call a target function with arguments"""
        super().register_all_targets()
        self.available_targets["zigzag"] = lambda prog, output: None
        self.available_targets["zigzag-workload"] = lambda prog, output: None


def main():
    praxis_main = PraxisMain()
    praxis_main.run()


if "__main__" == __name__:
    main()
