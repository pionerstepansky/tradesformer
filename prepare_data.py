import numpy as np
from tqdm import tqdm
import os
import pandas as pd


def find_next_trade_index(ts, trades, current_trade_index):
    '''
      Find last trade index with trade.ts <= ts condition.
    '''
    for i in range(current_trade_index, len(trades)):
        if trades.iloc[i].ts > ts:
            return i - 1
    return len(trades) - 1


def indexing(order_books, trades):
    current_trade_index = 0
    last_trade_indices = np.zeros(len(order_books))
    print('Indexing Start: ')
    for index in tqdm(range(len(order_books)), position=0, leave=True):
        current_ts = order_books.iloc[index].ts
        current_trade_index = find_next_trade_index(current_ts, trades, current_trade_index)
        last_trade_indices[index] = current_trade_index
    return last_trade_indices


def train_val_split(order_books, trades, targets, val_size):
    order_books_len = len(order_books)
    train_size = (1 - val_size)
    train_test_split_timestamp = order_books.iloc[int(order_books_len * train_size)].ts
    train_order_books = order_books[order_books.ts <= train_test_split_timestamp]
    val_order_books = order_books[order_books.ts > train_test_split_timestamp]
    train_trades = trades[trades.ts <= train_test_split_timestamp]
    val_trades = trades[trades.ts > train_test_split_timestamp]
    train_targets = targets.iloc[:len(train_order_books)]
    val_targets = targets.iloc[len(train_order_books):]
    print('Train data:')
    print(f'\tOrder books: {len(train_order_books)}')
    print(f'\tTrades: {len(train_trades)}')
    print('Val data:')
    print(f'\tOrder books: {len(val_order_books)}')
    print(f'\tTrades: {len(val_trades)}')
    return (train_order_books, val_order_books), (train_trades, val_trades), (train_targets, val_targets)


def prepare_data(order_books, trades, targets=None):
    prices_columns = [col for col in order_books.columns if 'price' in col]
    order_books_avg = order_books.loc[:, prices_columns].expanding().mean()
    order_books_std = order_books.loc[:, prices_columns].expanding().std().fillna(1)
    order_books.loc[:, prices_columns] = (order_books.loc[:, prices_columns] - order_books_avg) / order_books_std

    trades_avg = trades.loc[:, 'price'].expanding().mean()
    trades_std = trades.loc[:, 'price'].expanding().std().fillna(1)
    trades.loc[:, 'price'] = (trades.loc[:, 'price'] - trades_avg) / trades_std

    trades.loc[:, 'quantity'] = trades.loc[:, 'quantity']
    trades.loc[trades['side'] == 'buy', 'side'] = 1
    trades.loc[trades['side'] == 'sell', 'side'] = 0
    trades.loc[:, 'side'] = trades.loc[:, 'side'].astype('int')
    order_books['last_trade_idx'] = indexing(order_books, trades)
    order_books.fillna(0, inplace=True)
    trades.fillna(0, inplace=True)
    order_books.drop(columns=['ts'], inplace=True)
    trades.drop(columns=['ts'], inplace=True)
    if targets is not None:
        targets[targets == -1] = 0
    return order_books, trades, targets


def read_and_preprocess_data(data_path, is_train=True, val_size=None):
    order_books = pd.read_csv(os.path.join(data_path, 'order_books.csv'), index_col=[0])
    print(order_books.head(10))
    trades = pd.read_csv(os.path.join(data_path, 'trades.csv'), index_col=[0])
    if is_train:
        targets = pd.read_csv(os.path.join(data_path, 'targets.csv'), index_col=[0])
        if val_size is not None:
            (train_order_books, val_order_books), (train_trades, val_trades), (
                train_targets, val_targets) = train_val_split(order_books, trades, targets, val_size)
            train_order_books, train_trades, train_targets = prepare_data(train_order_books, train_trades,
                                                                          train_targets)
            val_order_books, val_trades, val_targets = prepare_data(val_order_books, val_trades, val_targets)
            return (train_order_books, val_order_books), (train_trades, val_trades), (train_targets, val_targets)
        else:
            order_books, trades, targets = prepare_data(order_books, trades, targets)
            return order_books, trades, targets
    else:
        return prepare_data(order_books, trades)
