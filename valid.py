import argparse
import os

import pandas as pd

from dataset import TradesDataset
from prepare_data import read_and_preprocess_data
import tensorflow as tf

# hyperparameters
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
    parser.add_argument('--val_size', type=int, default=None, choices=range(0, 101), metavar="[0-100]", help='Validate data size in percents. The value must be from 0 to 100.')
    args = parser.parse_args()

    if args.val_size:
        (train_order_books, val_order_books), (train_trades, val_trades), (
            train_targets, val_targets) = read_and_preprocess_data(args.dataset, args.val_size)
    else:
        (train_order_books, val_order_books), (train_trades, val_trades), (
            train_targets, val_targets) = read_and_preprocess_data(args.dataset)

    train_dataset = TradesDataset(train_order_books, train_trades, train_targets, batch_size=batch_size,
                                  order_books_seq_len=seq_len, trades_seq_len=seq_len)
    val_dataset = TradesDataset(val_order_books, val_trades, val_targets, batch_size=batch_size,
                                order_books_seq_len=seq_len, trades_seq_len=seq_len)

    model = create_model(batch_size, seq_len, d_k, d_v, n_heads, ff_dim)
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint('Tradesformer.hdf5',
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)

    history = model.fit(train_dataset,
                        epochs=3,
                        steps_per_epoch=1000,
                        validation_steps=100,
                        callbacks=[callback],
                        validation_data=val_dataset,
                        use_multiprocessing=True,
                        shuffle=True,
                        workers=num_workers)

