from ArgumentParser import arg_parser
from DominoParser import DominoParser

if __name__ == '__main__':
    argument_parser = arg_parser()
    env_args = vars(argument_parser.parse_args())
    parser = DominoParser(env_args['code'], env_args['iter'], reverse=env_args['reverse'])
    print(parser.parse())
