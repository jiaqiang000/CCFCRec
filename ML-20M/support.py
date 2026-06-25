import gc
import pickle
import random
import os
import sys
import time
import numpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


# 返回一个dict， userId: [positive_sample(list), negative_sample(list(list))]
def read_user_positive_negative_movies(user_positive_movie_csv, refresh=False):
    pkl_name = 'pkl/user_pn_dict.pkl'
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    if (os.path.exists(pkl_name) is True) and (refresh is False):
        pkl_file = open(pkl_name, 'rb')
        data = pickle.load(pkl_file)
        return data['user_pn_dict']
    user_position_dict = {}
    last_user = -1
    user_position_dict[last_user] = [-1, -1]
    for index, row in tqdm(user_positive_movie_csv.iterrows()):
        u = row['userId']
        if u != last_user:
            user_position_dict[u] = [index, index]
            user_position_dict[last_user] = [user_position_dict.get(last_user)[0], index-1]
            last_user = u
    # 更新最后一项
    user_position_dict[last_user] = [user_position_dict.get(last_user)[0], user_positive_movie_csv.__len__()-1]
    with open(pkl_name, 'wb') as file:
        pickle.dump({'user_pn_dict': user_position_dict}, file)
    return user_position_dict


def read_img_feature(img_feature_csv):
    df = pd.read_csv(img_feature_csv, dtype={'feature': object, 'movie_id': int})
    img_feature_dict = {}
    for index, row in df.iterrows():
        item = row['movie_id']
        feature = list(map(float, row['feature'][1:-1].split(",")))
        img_feature_dict[item] = feature
    return img_feature_dict


def read_genres(genres_csv):
    df = pd.read_csv(genres_csv, dtype={'movieId': int})
    genres_dict = {}
    for index, row in df.iterrows():
        item = row['movieId']
        genres = list(map(int, row['genres_onehot'][1:-1].split(',')))
        genres_dict[item] = genres
    return genres_dict


def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user原始下标，value: user有序下标
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, user_positive_movie_csv, img_features, genres, user_serialize_dict, item_pn_csv,
                 positive_number, negative_number):
        self.train_csv = train_csv
        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres
        self.user_pos_neg_movie_df = pd.read_csv(user_positive_movie_csv, dtype={'userId': np.int32, 'positive_movies': np.int32})
        self.item_pn_df = pd.read_csv(item_pn_csv)
        # print(self.item_pn_df)
        self.user_position_dict = read_user_positive_negative_movies(self.user_pos_neg_movie_df)
        self.user = self.train_csv["userId"]
        self.neg_user = self.train_csv['neg_user_id']
        self.item = self.train_csv["movieId"]
        self.rating = self.train_csv["rating"]
        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_item(self.item)
        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(self.item_serialize_dict)
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.user_values = self.user.to_numpy()
        self.item_values = self.item.to_numpy()
        self.neg_user_values = self.neg_user.to_numpy()
        self.serialized_user_values = np.asarray(
            [self.user_serialize_dict.get(user) for user in self.user_values],
            dtype=np.int64,
        )
        self.serialized_item_values = np.asarray(
            [self.item_serialize_dict.get(item) for item in self.item_values],
            dtype=np.int64,
        )
        self.serialized_neg_user_values = np.asarray(
            [self.user_serialize_dict.get(user) for user in self.neg_user_values],
            dtype=np.int64,
        )
        self.user_pos_positive_values = self.user_pos_neg_movie_df['positive_movies'].to_numpy()
        self.user_pos_negative_values = self.user_pos_neg_movie_df['negative_movies'].to_numpy()
        self.self_negative_values = self.item_pn_df['negative_movies'].to_numpy()
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def sample_serial_negatives_from_text(self, negative_movies):
        negative_items = np.fromstring(negative_movies[1:-1], sep=",", dtype=np.int64)
        negative_serial_items = np.asarray(
            [self.item_serialize_dict.get(item) for item in negative_items],
            dtype=np.int64,
        )
        return np.random.choice(negative_serial_items, self.negative_number, replace=True)

    def __getitem__(self, index):
        user = self.user_values[index]
        item = self.item_values[index]
        # 处理 item genres
        genres = self.genres_dict.get(item)
        # 处理 item feature
        img_feature = self.img_feature_dict.get(item)
        # 处理 positive items
        # 直接存储df的位置，user -> 从哪里到哪里
        position_arr = self.user_position_dict.get(user)
        positive_indices = np.random.randint(position_arr[0], position_arr[1] + 1, size=self.positive_number)
        positive_movie_list = self.user_pos_positive_values[positive_indices]
        negative_movie_list = self.user_pos_negative_values[positive_indices]
        self_negative_list = self.self_negative_values[index]
        # self neg list 完成 序列化
        self_neg_list = self.sample_serial_negatives_from_text(self_negative_list)
        # coll neg完成序列化
        neg_list = []
        for neg in negative_movie_list:
            # 插入一条抽样
            neg_list.append(self.sample_serial_negatives_from_text(neg))
        # 对当前item进行抽样
        # user，item id进行序列化
        user = self.serialized_user_values[index]
        neg_user = self.serialized_neg_user_values[index]
        item = self.serialized_item_values[index]
        # 序列化positive_movie_list
        positive_movie_list = [self.item_serialize_dict.get(item) for item in positive_movie_list]
        return torch.as_tensor(user, dtype=torch.long), torch.as_tensor(item, dtype=torch.long),\
               torch.as_tensor(genres, dtype=torch.long), torch.as_tensor(img_feature, dtype=torch.float32),\
               torch.as_tensor(neg_user, dtype=torch.long), torch.as_tensor(positive_movie_list, dtype=torch.long),\
               torch.as_tensor(np.asarray(neg_list), dtype=torch.long), torch.as_tensor(self_neg_list, dtype=torch.long)
