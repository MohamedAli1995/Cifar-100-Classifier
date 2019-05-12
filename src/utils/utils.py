import _pickle as cPickle
import argparse as arg


def get_args():
    argparse = arg.ArgumentParser(description=__doc__)

    argparse.add_argument(
        '-c', '--config',
        metavar='c',
        default='None',
        help='Config file path')

    argparse.add_argument(
        '-i', '--img_path',
        metavar='o',
        default='None',
        help='image path')
    args = argparse.parse_args()
    return args


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict
