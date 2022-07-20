import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Match program')
    parser.add_argument('-c', '--code', help='Match code', required=True)
    parser.add_argument('-i', '--iter', type=int, help='Iterations count', required=True)
    parser.add_argument('-r', '--reverse', action=argparse.BooleanOptionalAction, help='Reverse matching')
    return parser
