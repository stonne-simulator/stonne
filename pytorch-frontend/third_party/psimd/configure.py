#!/usr/bin/env python


import confu
parser = confu.standard_parser("FP16 configuration script")


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["psimd.h"])

    return build


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
