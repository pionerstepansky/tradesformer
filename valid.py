import argparse
import os

import pandas as pd

from dataset import TradesDataset
from model import create_model
from prepare_data import read_and_preprocess_data, prepare_data
import tensorflow as tf
from contants import *
from sklearn.model_selection import KFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Input directory. Must include order_books.csv, trades.csv, targets.csv')
    parser.add_argument('--folds', type=int, default=2, choices=range(0, 10), metavar="[0-10]", help='Folds count.')
    args = parser.parse_args()

    order_books, trades, targets = read_and_preprocess_data(args.dataset)
    kf = KFold(n_splits=args.folds)
    current_fold = 0
    for train_index, val_index in kf.split(order_books):
        print('-----------------------------------------------------')
        print(f'Start fold {current_fold}:')
        train_order_books, val_order_books = order_books.iloc[train_index], order_books.iloc[val_index]
        train_targets, val_targets = order_books.iloc[train_index], order_books.iloc[val_index]
        # left_train_timestamp = train_order_books.iloc[0].ts
        # right_train_timestamp = train_order_books.iloc[len(train_order_books) - 1].ts
        # train_trades = trades[left_train_timestamp < trades.ts <= right_train_timestamp]
        # test_trades = pd.concat([trades[trades.ts < left_train_timestamp], trades[right_train_timestamp < trades.ts]])
        train_dataset = TradesDataset(train_order_books, trades, train_targets, batch_size=batch_size,
                                      order_books_seq_len=seq_len, trades_seq_len=seq_len)
        val_dataset = TradesDataset(val_order_books, trades, val_targets, batch_size=batch_size,
                                    order_books_seq_len=seq_len, trades_seq_len=seq_len)
        model = create_model(batch_size, seq_len, d_k, d_v, n_heads, ff_dim)
        model.fit(train_dataset,
                  epochs=1,
                  validation_data=val_dataset,
                  use_multiprocessing=True,
                  shuffle=True,
                  workers=num_workers)
        current_fold += 1
        print(f'fold {current_fold} Finished')

