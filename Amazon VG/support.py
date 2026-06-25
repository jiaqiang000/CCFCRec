import random
import time
import math

from torch.utils.data import Dataset
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle


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


def sample_negative_user(user_set, interaction_user_set):
    users = set(interaction_user_set)
    candidate_users = set(user_set) - set(users)
    return random.sample(list(candidate_users), 1)[0]


def _rng_integers(rng, high, size):
    if hasattr(rng, "integers"):
        return rng.integers(0, high, size=size)
    return rng.randint(0, high, size=size)


def sample_negative_serial_items(item_number, excluded_items, sample_size, rng=np.random):
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if item_number <= 0:
        raise ValueError("item_number must be positive")
    if sample_size == 0:
        return np.empty(0, dtype=np.int64)

    excluded_set = {
        int(item)
        for item in excluded_items
        if 0 <= int(item) < item_number
    }
    candidate_count = item_number - len(excluded_set)
    if candidate_count <= 0:
        raise ValueError("no negative item candidate exists")

    if candidate_count < item_number * 0.2:
        candidates = np.fromiter(
            (item for item in range(item_number) if item not in excluded_set),
            dtype=np.int64,
        )
        candidate_indices = _rng_integers(rng, len(candidates), sample_size)
        return candidates[candidate_indices]

    samples = np.empty(sample_size, dtype=np.int64)
    filled = 0
    draw_size = max(sample_size, 64)
    while filled < sample_size:
        draw = _rng_integers(rng, item_number, draw_size)
        if excluded_set:
            draw = np.fromiter(
                (int(item) for item in draw if int(item) not in excluded_set),
                dtype=np.int64,
            )
        take = min(sample_size - filled, len(draw))
        if take > 0:
            samples[filled:filled + take] = draw[:take]
            filled += take
    return samples


def compute_history_category_support_confidence(
    current_item,
    history_items,
    item_category_sets,
    adaptive_history_max_count=20,
):
    item_categories = item_category_sets.get(current_item, frozenset())
    if len(item_categories) == 0:
        return 0.0
    other_history_items = [item for item in history_items if item != current_item]
    if len(other_history_items) == 0:
        return 0.0
    history_categories = set()
    for history_item in other_history_items:
        history_categories.update(item_category_sets.get(history_item, frozenset()))
    if len(history_categories) == 0:
        return 0.0
    category_overlap = len(item_categories.intersection(history_categories)) / len(item_categories)
    history_len_conf = min(
        math.log1p(len(other_history_items)) / math.log1p(adaptive_history_max_count),
        1.0,
    )
    return float(category_overlap * history_len_conf)


# 新建一个user-item的交互字典
def build_user_item_interaction_dict(train_csv='data/train_rating.csv',
                                     user_item_interaction_dict_save='pkl/user_item_interaction_dict.pkl'):
    if os.path.exists(user_item_interaction_dict_save) is True:
        print('从缓存中加载user_item_interaction_dict')
        pkl_file = open(user_item_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['user_item_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    user_item_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = user_item_interaction_dict.get(user)
        if res is None:
            user_item_interaction_dict[user] = [movie]
        else:
            res.append(movie)
            user_item_interaction_dict[user] = res
    with open(user_item_interaction_dict_save, 'wb') as file:
        pickle.dump({'user_item_interaction_dict': user_item_interaction_dict}, file)
    return user_item_interaction_dict


# 新建一个item-user的交互字典
def build_item_user_interaction_dict(train_csv='data/train_rating.csv',
                                     item_user_interaction_dict_save='pkl/item_user_interaction_dict.pkl'):
    if os.path.exists(item_user_interaction_dict_save) is True:
        print('从缓存中加载', item_user_interaction_dict_save)
        pkl_file = open(item_user_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['item_user_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    item_user_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = item_user_interaction_dict.get(movie)
        if res is None:
            item_user_interaction_dict[movie] = [user]
        else:
            res.append(user)
            item_user_interaction_dict[movie] = res
    with open(item_user_interaction_dict_save, 'wb') as file:
        pickle.dump({'item_user_interaction_dict': item_user_interaction_dict}, file)
    return item_user_interaction_dict


class RatingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_csv,
        img_features,
        genres,
        category_num,
        user_serialize_dict,
        positive_number,
        negative_number,
        adaptive_history_max_count=20,
    ):
        self.train_csv = train_csv
        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres
        # print(self.item_pn_df)
        self.user = self.train_csv["reviewerID"]
        self.item = self.train_csv["asin"]
        self.rating = self.train_csv["rating"]
        self.neg_user = self.train_csv['neg_user']
        self.item_set = set(self.item)
        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_item(self.item)
        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(self.item_serialize_dict)
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.adaptive_history_max_count = adaptive_history_max_count
        self.category_num = category_num
        self.user_item_interaction_dict = build_user_item_interaction_dict()
        self.item_user_interaction_dict = build_item_user_interaction_dict()
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
        self.user_positive_serial_items = {}
        self.user_positive_serial_sets = {}
        for user, items in self.user_item_interaction_dict.items():
            serial_items = np.asarray(
                [
                    self.item_serialize_dict[item]
                    for item in items
                    if item in self.item_serialize_dict
                ],
                dtype=np.int64,
            )
            if len(serial_items) == 0:
                continue
            self.user_positive_serial_items[user] = serial_items
            self.user_positive_serial_sets[user] = set(serial_items.tolist())
        self.item_category_sets = {
            item: frozenset(int(category) for category in categories)
            for item, categories in self.genres_dict.items()
        }
        self.support_confidence_dict = self.build_support_confidence_dict()
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def build_support_confidence_dict(self):
        support_confidence_dict = {}
        for user, item in zip(self.user, self.item):
            key = (user, item)
            if key in support_confidence_dict:
                continue
            history_items = self.user_item_interaction_dict.get(user, [])
            support_confidence_dict[key] = compute_history_category_support_confidence(
                current_item=item,
                history_items=history_items,
                item_category_sets=self.item_category_sets,
                adaptive_history_max_count=self.adaptive_history_max_count,
            )
        return support_confidence_dict

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        user = self.user_values[index]
        item = self.item_values[index]
        support_confidence = self.support_confidence_dict.get((user, item), 0.0)
        # 处理 item genres
        genres = torch.full((self.category_num,), -1)
        genres_index = self.genres_dict.get(item)
        genres[genres_index] = 1
        # 处理 item feature
        img_feature = self.img_feature_dict.get(item)
        # --------------------- #
        #  处理 positive items   #
        #  runtime sampling     #
        # --------------------- #
        positive_items_ = self.user_positive_serial_items.get(user)
        if positive_items_ is None:
            positive_items_ = np.asarray([self.serialized_item_values[index]], dtype=np.int64)
        positive_items_list = np.random.choice(positive_items_, self.positive_number, replace=True)
        # runtime sampling negative
        positive_item_set = self.user_positive_serial_sets.get(user, set(positive_items_.tolist()))
        negative_items_ = sample_negative_serial_items(
            self.item_number,
            positive_item_set,
            self.negative_number*(self.positive_number+1),
        )
        negative_item_list = negative_items_[:self.negative_number*self.positive_number].reshape(
            self.positive_number,
            self.negative_number,
        )
        # self neg list 完成 序列化, self的抽样放在和collaborative items中一起抽样负例子，最后分割出来就行了
        self_neg_list = negative_items_[self.positive_number*self.negative_number:]
        # serialize
        user = self.serialized_user_values[index]
        item = self.serialized_item_values[index]
        neg_user = self.serialized_neg_user_values[index]
        return torch.as_tensor(user, dtype=torch.long), torch.as_tensor(item, dtype=torch.long), genres,\
               torch.as_tensor(img_feature, dtype=torch.float32), torch.as_tensor(neg_user, dtype=torch.long),\
               torch.as_tensor(positive_items_list, dtype=torch.long), torch.as_tensor(negative_item_list, dtype=torch.long),\
               torch.as_tensor(self_neg_list, dtype=torch.long),\
               torch.tensor(support_confidence, dtype=torch.float32)


# 测试数据封装
if __name__ == '__main__':
    print("support.py")

    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    category_length = len(category_ser_map)
    # (self, train_csv, img_features, genres, user_serialize_dict, positive_number, negative_number)
    train_csv = pd.read_csv("data/train_withneg_rating.csv")
    print("ratings.length:", train_csv.__len__())
    all_ratings = pd.read_csv("data/ratings_filter.csv")
    user_ser_dict = serialize_user(all_ratings["reviewerID"])
    dataset = RatingDataset(train_csv, img_feature_dict, asin_category_int_map, category_length, user_ser_dict, 10, 20)
    dataIter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    it = dataIter.__iter__()
    for i_index in range(10):
        start = time.time()
        u, i, g, i_f, n_user, p_list, n_list, self_n_list, support_confidence = it.next()
        print("time spend:", time.time()-start)
        i_index += 1
    # print(u, i, g, i_f, n_user)
    # print("positive_list, negative_list, self_negative_list")
    # print("genres.shape:", g.shape, "img_f.shape:", i_f.shape)
