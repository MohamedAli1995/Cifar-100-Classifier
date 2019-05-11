import argparse

def get_args():
    argparse = argparse.ArgumentParser(description=__doc__)
    argparse.add_argument(
        '-c', '--config',
        metavar='c',
        default='None',
        help='Config file path')
    args = argparse.parse_args()
    return args