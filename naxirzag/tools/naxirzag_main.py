from xdsl.xdsl_opt_main import xDSLOptMain

from naxirzag.transforms import get_all_passes


class NaxirzagMain(xDSLOptMain):
    def register_all_passes(self):
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)


def main():
    naxirzag_main = NaxirzagMain()
    naxirzag_main.run()


if "__main__" == __name__:
    main()
