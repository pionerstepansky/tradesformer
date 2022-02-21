import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import TradesDataset
from layers import CUSTOM_LAYERS
from prepare_data import read_and_preprocess_data
from train import СHECKPOINT


# hyperparameters
batch_size = 1
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

    test_order_books, test_trades, _ = read_and_preprocess_data(args.dataset, is_train=False)

    padded_test_order_books = pd.DataFrame(np.pad(test_order_books.values, ((seq_len - 1, 0), (0, 0)),  'edge'),
                                    columns=test_order_books.columns)

    padded_test_trades = pd.DataFrame(np.pad(test_trades.values, ((seq_len - 1, 0), (0, 0)),  'edge'),
                                    columns=test_trades.columns)

    test_dataset = TradesDataset(padded_test_order_books, padded_test_trades, targets=None, batch_size=batch_size,
                                 order_books_seq_len=seq_len, trades_seq_len=seq_len)

    model = tf.keras.models.load_model(СHECKPOINT, custom_objects=CUSTOM_LAYERS)

    probs = model.predict(test_dataset)
    prediction = (probs > 0.5).astype('int')
    prediction[prediction == 0] = -1
    prediction = pd.Series(prediction[:, 0], name='target')
    assert len(prediction) == len(test_order_books)
    prediction.to_csv('prediction.csv')

