import argparse
import os

import pandas as pd

from dataset import TradesDataset
from contants import *
from model import create_model
from prepare_data import read_and_preprocess_data
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Input directory. Must include order_books.csv, trades.csv.')
    parser.add_argument('--val_size', type=int, default=None, choices=range(0, 101), metavar="[0-100]",
                        help='Validate data size in percents. The value must be from 0 to 100.')
    parser.add_argument('--epoch', type=int, default=1,  help='Number of training epochs')
    args = parser.parse_args()

    if args.val_size:
        (train_order_books, val_order_books), (train_trades, val_trades), (
            train_targets, val_targets) = read_and_preprocess_data(args.dataset, is_train=True,
                                                                   val_size=args.val_size / 100)
    else:
        (train_order_books, val_order_books), (train_trades, val_trades), (
            train_targets, val_targets) = read_and_preprocess_data(args.dataset)

    train_dataset = TradesDataset(train_order_books, train_trades, train_targets, batch_size=batch_size,
                                  order_books_seq_len=seq_len, trades_seq_len=seq_len)
    val_dataset = TradesDataset(val_order_books, val_trades, val_targets, batch_size=batch_size,
                                order_books_seq_len=seq_len, trades_seq_len=seq_len)

    model = create_model(batch_size, seq_len, d_k, d_v, n_heads, ff_dim)
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(СHECKPOINT,
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    history = model.fit(train_dataset,
                        epochs=args.epoch,
                        callbacks=[tensorboard, checkpoint],
                        validation_data=val_dataset,
                        use_multiprocessing=True,
                        shuffle=True,
                        workers=num_workers)

    print('Training finished')
    print(f'Model saved to {СHECKPOINT}')
