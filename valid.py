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

    # order_books, trades, targets = read_and_preprocess_data(args.dataset)
    order_books = pd.read_csv(os.path.join(args.dataset, 'order_books.csv'), index_col=[0])
    trades = pd.read_csv(os.path.join(args.dataset, 'trades.csv'), index_col=[0])
    targets = pd.read_csv(os.path.join(args.dataset, 'targets.csv'), index_col=[0])

    kf = KFold(n_splits=args.folds)
    current_fold = 0
    for train_index, test_index in kf.split(order_books):
        print('-----------------------------------------------------')
        print(f'Start fold {current_fold}:')
        train_order_books, test_order_books = order_books.iloc[train_index], order_books.iloc[test_index]
        train_targets, test_targets = order_books.iloc[train_index], order_books.iloc[test_index]
        left_train_timestamp = train_order_books.iloc[0].ts
        right_train_timestamp = train_order_books.iloc[len(train_order_books) - 1].ts
        # train_trades = trades[left_train_timestamp < trades.ts <= right_train_timestamp]
        # test_trades = pd.concat([trades[trades.ts < left_train_timestamp], trades[right_train_timestamp < trades.ts]])
        train_order_books, train_trades, train_targets = prepare_data(train_order_books, trades, train_targets)
        val_order_books, val_trades, val_targets = prepare_data(train_order_books, trades, train_targets)
        train_dataset = TradesDataset(train_order_books, train_trades, train_targets, batch_size=batch_size,
                                      order_books_seq_len=seq_len, trades_seq_len=seq_len)
        val_dataset = TradesDataset(val_order_books, val_trades, val_targets, batch_size=batch_size,
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

