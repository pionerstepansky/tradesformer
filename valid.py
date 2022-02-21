import argparse
import os

import pandas as pd

from dataset import TradesDataset
from prepare_data import read_and_preprocess_data
import tensorflow as tf
from contants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Input directory. Must include order_books.csv, trades.csv, targets.csv')
    parser.add_argument('--folds', type=int, default=None, choices=range(0, 10), metavar="[0-10]", help='Folds count.')
    args = parser.parse_args()

    order_books, trades, targets = read_and_preprocess_data(args.dataset)

