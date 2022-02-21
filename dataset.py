import concurrent
import math

import numpy as np
from tensorflow.keras.utils import Sequence


class TradesDataset(Sequence):

    def __init__(self, order_books, trades, targets=None, batch_size=32, order_books_seq_len=1, trades_seq_len=1,
                 shuffle=False):
        # pad for first sequences.
        super(TradesDataset, self).__init__()
        self.length = (len(order_books) - order_books_seq_len + 1)
        self.indexes = np.arange(self.length)
        self.order_books = order_books
        self.targets = targets
        self.trades = trades
        self.order_books_seq_len = order_books_seq_len
        self.trades_seq_len = trades_seq_len
        self.order_books_features_count = order_books.shape[1]
        self.trades_features_count = trades.shape[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.isTrain = targets is not None

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __get_sequence__(self, index):
        right_border = min(index + self.order_books_seq_len, len(self.order_books))
        order_books_seq = self.order_books.iloc[index:right_border]
        if self.isTrain:
            target = self.targets.iloc[right_border - 1]
        else:
            target = None
        trade_index = int(order_books_seq.iloc[self.order_books_seq_len - 1]['last_trade_idx'])
        if trade_index < 0:
            trades_seq = np.zeros((self.trades_seq_len, self.trades_features_count))
        else:
            trades_seq = self.trades.iloc[max(0, trade_index - self.trades_seq_len + 1): trade_index + 1].values
            if len(trades_seq) < self.trades_seq_len:
                trades_seq = np.pad(trades_seq, ((self.trades_seq_len - len(trades_seq), 0), (0, 0)), 'edge')
        order_books_seq.drop(columns=['last_trade_idx'], inplace=True)
        return order_books_seq.values, trades_seq, target

    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        '''
          Returns tuple (order_books_seq, trades_seq, target)
        '''
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        order_books, trades, targets = [], [], []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.__get_sequence__, idx) for idx in indexes]
            for fut in futures:
                order_books_seq, trades_seq, target = fut.result()
                order_books.append(order_books_seq)
                trades.append(trades_seq)
                targets.append(target)
            if self.isTrain:
                return [np.array(order_books), np.array(trades)], np.array(targets, dtype='float').squeeze()
            else:
                return [np.array(order_books), np.array(trades)]
