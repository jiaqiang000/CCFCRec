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


SUPPORTED_NEGATIVE_SAMPLING_MODES = {"legacy_cached", "fast_uniform", "original_np_choice"}


def sample_original_np_choice_items(
    item_set,
    item_serialize_dict,
    positive_raw_items,
    positive_number,
    negative_sample_size,
    rng=np.random,
):
    """复刻 2026-06-08/09 实验使用的原始 raw-item 负采样路径。

    这个函数故意不做 compact-index/rejection sampling，也不排序候选集合。诊断目的就是
    让负采样“世界”回到旧代码：
    `np.random.choice(list(self.item_set - set(positive_items_)), replace=True)`。
    因此这里保留 Python set 转 list 的候选顺序、raw item -> serial id 的后置映射，
    用来隔离最近负采样优化是否改变了 baseline/category_conf_input 的相对结果。
    """
    if positive_number < 0:
        raise ValueError("positive_number must be non-negative")
    if negative_sample_size < 0:
        raise ValueError("negative_sample_size must be non-negative")

    positive_candidates = list(positive_raw_items)
    if len(positive_candidates) == 0:
        raise ValueError("positive_raw_items must not be empty")
    positive_items = list(rng.choice(positive_candidates, positive_number, replace=True))
    positive_serial_items = np.asarray(
        [item_serialize_dict[item] for item in positive_items],
        dtype=np.int64,
    )

    negative_candidates = list(set(item_set) - set(positive_raw_items))
    if len(negative_candidates) == 0:
        raise ValueError("no negative item candidate exists")
    negative_items = list(rng.choice(negative_candidates, negative_sample_size, replace=True))
    negative_serial_items = np.asarray(
        [item_serialize_dict[item] for item in negative_items],
        dtype=np.int64,
    )
    return positive_serial_items, negative_serial_items


class LegacyCachedNegativeSampler:
    """用 compact-index 方式保留旧协议候选语义，同时避免构造完整候选数组。

    改动原因：
    cfd64c2 中的 `sample_negative_serial_items` 直接在序列化 item id 空间做 rejection
    sampling，速度快，但它改变了旧代码 `np.random.choice(list(self.item_set -
    set(positive_items_)))` 的候选集合构造顺序和随机路径。对固定 seed 的训练来说，这会让
    负样本序列变化，从而改变训练轨迹。

    这个类保留旧协议的核心语义：每个 user 的负候选仍然是
    `训练 item 全集 - 该 user 的训练交互 item`。性能优化不再缓存完整负候选数组，因为
    Amazon-VG 有 5 万多个 user，shuffle 后小 LRU cache 命中率极低，会退回到反复构造
    2 万多个 item 候选数组的慢路径。

    当前实现只为每个 user 保存“需要跳过的正样本 serial id”。采样时先在 compact 候选
    空间 `[0, item_number - positive_count)` 里抽索引，再把索引平移到跳过正样本后的
    serial id。这样不生成大候选列表，随机数消耗也比 rejection sampling 更接近旧的
    `np.random.choice(候选列表)`。
    """

    def __init__(self, item_set, item_serialize_dict, user_item_interaction_dict, max_cache_size=512):
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        self.item_number = len(item_serialize_dict)
        self.user_positive_serial_items = {}
        for user, positive_items in user_item_interaction_dict.items():
            positive_serial_items = sorted(
                {
                    item_serialize_dict[item]
                    for item in positive_items
                    if item in item_serialize_dict
                }
            )
            if positive_serial_items:
                self.user_positive_serial_items[user] = np.asarray(positive_serial_items, dtype=np.int64)

    def _map_compact_indices_to_serial_items(self, compact_indices, positive_serial_items):
        # compact_indices 是“移除正样本后的候选列表”下标。下面按正样本 serial id 的位置
        # 逐个跳过空洞，把 compact 下标映射回完整 serial id 空间。
        # 用户正样本数量通常很小，循环次数远少于 item_number，因此比构造完整候选数组快很多。
        serial_items = compact_indices.astype(np.int64, copy=True)
        for positive_serial_item in positive_serial_items:
            serial_items += serial_items >= positive_serial_item
        return serial_items

    def sample(self, user, sample_size, rng=np.random):
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if sample_size == 0:
            return np.empty(0, dtype=np.int64)
        positive_serial_items = self.user_positive_serial_items.get(user)
        if positive_serial_items is None or positive_serial_items.size == 0:
            return _rng_integers(rng, self.item_number, sample_size).astype(np.int64, copy=False)
        candidate_count = self.item_number - positive_serial_items.size
        if candidate_count <= 0:
            raise ValueError(f"no negative item candidate exists for user {user}")
        compact_indices = _rng_integers(rng, candidate_count, sample_size)
        return self._map_compact_indices_to_serial_items(compact_indices, positive_serial_items)


def build_item_feature_tensors(item_serialize_dict, img_features, genres, category_num):
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if category_num <= 0:
        raise ValueError("category_num must be positive")

    item_count = len(item_serialize_dict)
    ordered_items = sorted(item_serialize_dict.items(), key=lambda item_pair: item_pair[1])
    first_item = ordered_items[0][0]
    if first_item not in img_features:
        raise KeyError(f"missing image feature for item {first_item}")
    first_image = torch.as_tensor(img_features[first_item], dtype=torch.float32)
    if first_image.ndim != 1:
        raise ValueError("image features must be one-dimensional")

    category_tensor = torch.full((item_count, category_num), -1, dtype=torch.int8)
    image_tensor = torch.empty((item_count, first_image.numel()), dtype=torch.float32)

    for raw_item, serial_item in ordered_items:
        if serial_item < 0 or serial_item >= item_count:
            raise ValueError(f"serialized item id out of range for item {raw_item}: {serial_item}")
        image_feature = img_features.get(raw_item)
        if image_feature is None:
            raise KeyError(f"missing image feature for item {raw_item}")
        image_feature = torch.as_tensor(image_feature, dtype=torch.float32)
        if image_feature.ndim != 1 or image_feature.numel() != first_image.numel():
            raise ValueError(f"inconsistent image feature shape for item {raw_item}")
        image_tensor[serial_item] = image_feature

        genre_indices = genres.get(raw_item)
        if genre_indices is None:
            continue
        genre_indices = torch.as_tensor(list(genre_indices), dtype=torch.long)
        if genre_indices.numel() == 0:
            continue
        category_tensor[serial_item, genre_indices] = 1

    return category_tensor, image_tensor


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
        negative_sampling_mode="legacy_cached",
        negative_sampling_cache_size=512,
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
        self.negative_sampling_mode = negative_sampling_mode
        self.negative_sampling_cache_size = negative_sampling_cache_size
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
        if self.negative_sampling_mode not in SUPPORTED_NEGATIVE_SAMPLING_MODES:
            raise ValueError(f"unsupported negative_sampling_mode={self.negative_sampling_mode}")
        self.legacy_negative_sampler = LegacyCachedNegativeSampler(
            item_set=self.item_set,
            item_serialize_dict=self.item_serialize_dict,
            user_item_interaction_dict=self.user_item_interaction_dict,
            max_cache_size=self.negative_sampling_cache_size,
        )
        self.item_category_tensor, self.item_image_feature_tensor = build_item_feature_tensors(
            item_serialize_dict=self.item_serialize_dict,
            img_features=self.img_feature_dict,
            genres=self.genres_dict,
            category_num=self.category_num,
        )
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
        # --------------------- #
        #  处理 positive items   #
        #  runtime sampling     #
        # --------------------- #
        negative_sample_size = self.negative_number*(self.positive_number+1)
        if self.negative_sampling_mode == "original_np_choice":
            # 诊断模式：完整保留 6 月 8/9 号实验的 raw item 采样路径。
            # 注意这里故意不用预序列化正样本、不用 compact-index，也不排序 set-diff 候选。
            # 它会比当前优化采样慢，但可以隔离“最近负采样改动”是否导致实验地基漂移。
            positive_raw_items = self.user_item_interaction_dict.get(user)
            if positive_raw_items is None:
                positive_raw_items = [item]
            positive_items_list, negative_items_ = sample_original_np_choice_items(
                item_set=self.item_set,
                item_serialize_dict=self.item_serialize_dict,
                positive_raw_items=positive_raw_items,
                positive_number=self.positive_number,
                negative_sample_size=negative_sample_size,
            )
        else:
            positive_items_ = self.user_positive_serial_items.get(user)
            if positive_items_ is None:
                positive_items_ = np.asarray([self.serialized_item_values[index]], dtype=np.int64)
            positive_items_list = np.random.choice(positive_items_, self.positive_number, replace=True)
        # runtime sampling negative
        if self.negative_sampling_mode == "legacy_cached":
            # 正式实验默认使用 legacy_cached：候选集合语义回到优化前协议，但通过缓存
            # 避免每条样本重复构造大候选列表，降低 CPU 瓶颈。
            negative_items_ = self.legacy_negative_sampler.sample(user, negative_sample_size)
        elif self.negative_sampling_mode == "fast_uniform":
            # fast_uniform 保留 cfd64c2 的最快路径。它直接在序列化 item id 空间做
            # rejection sampling，速度更高，但会改变固定 seed 下的负样本序列；因此只作为
            # 显式选择的极限速度协议，不作为默认正式实验协议。
            positive_item_set = self.user_positive_serial_sets.get(user, set(positive_items_.tolist()))
            negative_items_ = sample_negative_serial_items(
                self.item_number,
                positive_item_set,
                negative_sample_size,
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
        return torch.as_tensor(user, dtype=torch.long), torch.as_tensor(item, dtype=torch.long),\
               torch.as_tensor(neg_user, dtype=torch.long),\
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
