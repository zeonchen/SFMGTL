import torch
import numpy as np
import random
import pickle
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfTransformer


def norm_adj(adj):
    node_num = adj.shape[0]
    row, col = np.diag_indices_from(adj)
    adj[row, col] = 0

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                adj[i, j] = 0

    adj = adj / (np.sum(adj, axis=0) + 1e-6)

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                adj[i, j] = 1

    return adj


def split_x_y(data, lag, val_num=60 * 24, test_num=60 * 24, pred_lag=0):
    train_x = []
    train_y = []
    train_t = []
    val_x = []
    val_y = []
    val_t = []
    test_x = []
    test_y = []
    test_t = []
    num_samples = int(data.shape[0])
    time_idx = pd.date_range('1/1/2016', '1/1/2017', freq='h')[:-1].to_series()
    dayofweek = time_idx.dt.dayofweek.values
    dayofweek = np.eye(np.max(dayofweek) + 1)[dayofweek]
    hour = time_idx.dt.hour.values
    hour = np.eye(np.max(hour) + 1)[hour]
    for i in range(-int(min(lag)), num_samples):
        if pred_lag == -1:
            x_idx = [int(_ + i) for _ in lag][:7*24]
        else:
            x_idx = [int(_ + i) for _ in lag][:6]
        y_idx = [i]
        x_ = data[x_idx, :, :]
        time_feat = np.concatenate([dayofweek[i, :], hour[i, :]]).reshape(1, -1)
        y_ = data[y_idx, :, :]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
            train_t.append(time_feat)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
            val_t.append(time_feat)
        else:
            test_x.append(x_)
            test_y.append(y_)
            test_t.append(time_feat)
    return np.stack(train_x, axis=0), np.stack(train_y, axis=0), np.stack(train_t, axis=0),\
           np.stack(val_x, axis=0), np.stack(val_y, axis=0), np.stack(val_t, axis=0),\
           np.stack(test_x, axis=0), np.stack(test_y, axis=0), np.stack(test_t, axis=0),


def min_max_normalize(data, percentile = 0.999):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl) * percentile)]
    min_val = max(0, sl[0])
    data[data > max_val] = max_val
    data -= min_val
    data /= (max_val - min_val)
    return data, max_val, min_val


def get_loader(city='NY', type='pickup', used_day=7, pred_lag=1):
    data = np.load("data/%s/%s%s_%s.npy" % (city, 'Bike', city, type))
    norm_data, max_, min_ = min_max_normalize(data)
    lng, lat = norm_data.shape[1], norm_data.shape[2]
    mask = norm_data.sum(0) > 0
    mask = torch.Tensor(mask.reshape(1, lng, lat))

    if pred_lag == 5:
        lag = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
    elif pred_lag == 3:
        lag = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    elif pred_lag == -1:
        lag = [i for i in range(-169, -2)]
    else:
        lag = [-6, -5, -4, -3, -2, -1]
    train_x, train_y, train_t, val_x, val_y, val_t, test_x, test_y, test_t = split_x_y(norm_data, lag, pred_lag=-1)
    x = np.concatenate([train_x, val_x, test_x], axis=0)
    y = np.concatenate([train_y, val_y, test_y], axis=0)
    t = np.concatenate([train_t, val_t, test_t], axis=0)

    if city == 'DC':
        train_x = train_x[-used_day * 24:, :, :, :]
        train_y = train_y[-used_day * 24:, :, :, :]
        train_t = train_t[-used_day * 24:, :, :]

        test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y), torch.Tensor(test_t))
        val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y), torch.Tensor(val_t))
        dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y), torch.Tensor(train_t))
    else:
        test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y), torch.Tensor(test_t))
        val_dataset = None
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y), torch.Tensor(t))

    poi = np.load("data/%s/%s_poi.npy" % (city, city))
    poi = poi.reshape(lng * lat, -1)  # regions * classes
    transform = TfidfTransformer()
    norm_poi = np.array(transform.fit_transform(poi).todense())

    prox_adj = np.load("data/%s/%s_prox_adj.npy" % (city, city))
    road_adj = np.load("data/%s/%s_road_adj.npy" % (city, city))
    poi_adj = np.load("data/%s/%s_poi_adj.npy" % (city, city))

    prox_adj = norm_adj(prox_adj)
    road_adj = norm_adj(road_adj)
    poi_adj = norm_adj(poi_adj)

    return dataset, val_dataset, test_dataset, mask, max_, min_, norm_poi, \
           prox_adj, road_adj, poi_adj

