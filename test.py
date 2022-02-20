import argparse
import os

import pandas as pd

from dataset import TradesDataset
from model import create_model
from prepare_data import read_and_preprocess_data
import tensorflow as tf

# hyperparameters
from train import СHECKPOINT

batch_size = 64
seq_len = 100
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
num_workers = 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Input directory. Must include order_books.csv, trades.csv.')
    args = parser.parse_args()

    (train_order_books, val_order_books), (train_trades, val_trades),  = read_and_preprocess_data(args.dataset)

    test_dataset = TradesDataset(train_order_books, train_trades, train_targets, batch_size=batch_size,
                                  order_books_seq_len=seq_len, trades_seq_len=seq_len)

    model = tf.keras.models.load_model(СHECKPOINT)

    model.predict(x)

