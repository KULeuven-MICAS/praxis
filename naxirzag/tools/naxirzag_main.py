from xdsl.xdsl_opt_main import xDSLOptMain

class NaxirzagMain(xDSLOptMain): ...

def main():
    naxirzag_main = NaxirzagMain()
    naxirzag_main.run()


if "__main__" == __name__:
    main()
