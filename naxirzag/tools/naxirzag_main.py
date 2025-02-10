from snaxc.tools.snax_opt_main import SNAXOptMain

from naxirzag.transforms import get_all_passes


class NaxirzagMain(SNAXOptMain):
    def register_all_passes(self):
        # Register all snax-opt passes (which includes all xdsl passes)
        super().register_all_passes()
        # Aditionally, register all naxirzag passes
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)


def main():
    naxirzag_main = NaxirzagMain()
    naxirzag_main.run()


if "__main__" == __name__:
    main()
