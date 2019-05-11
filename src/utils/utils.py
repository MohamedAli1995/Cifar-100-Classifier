import argparse
import _pickle as cPickle


def get_args():
    argparse = argparse.ArgumentParser(description=__doc__)
    argparse.add_argument(
        '-c', '--config',
        metavar='c',
        default='None',
        help='Config file path')
    args = argparse.parse_args()
    return args


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict
