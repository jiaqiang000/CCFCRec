import random

import pandas as pd
import torch
import numpy as np
import time
import os
from myargs import get_args
from tqdm import tqdm
from support import build_item_feature_tensors


def resolve_device():
    requested = os.environ.get('CCFCREC_DEVICE', '').strip().lower()
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if requested == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if requested:
        raise RuntimeError(f"Unsupported or unavailable CCFCREC_DEVICE={requested}")
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = resolve_device()
if device.type == 'cuda':
    torch.cuda.set_device(int(os.environ.get('CCFCREC_CUDA_DEVICE', '0')))


def get_random_user_rank_list(model, genres, image_feature, k):
    user_number = model.user_embedding.shape[0]
    user_list = list(range(0, user_number))
    res_list = []
    for i in range(k):
        res_list.append(random.sample(user_list, 1)[0])
    return res_list


def get_similar_user_speed(model, genres, image_feature, k):
    return get_similar_user_batch(
        model,
        genres.unsqueeze(dim=0),
        image_feature.unsqueeze(dim=0),
        k,
    )[0]


def get_similar_user_batch(model, genres, image_feature, k):
    q_v_c = model(genres, image_feature, genres.shape[0])
    user_emb = model.user_embedding
    ratings = torch.matmul(q_v_c, user_emb.t())
    top_k = min(k, ratings.shape[1])
    index = torch.topk(ratings, k=top_k, dim=1).indices
    return index.cpu().detach().numpy().tolist()


def hr_at_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    inter = set(groundtruth_user).intersection(set(recommend_users))
    return len(inter)


def dcg_k(r):
    r = np.asarray(r)
    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(1+1, r.size + 2))))
    return val


def ndcg_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    ratings = []
    ndcg = 0.0
    for u in recommend_users:
        if u in groundtruth_user:
            ratings.append(1.0)
        else:
            ratings.append(0.0)
    ratings_ideal = sorted(ratings, reverse=True)
    ideal_dcg = dcg_k(ratings_ideal)
    if ideal_dcg != 0:
        ndcg = (dcg_k(ratings) / ideal_dcg)
    return ndcg


class Validate:
    def __init__(self, validate_csv, user_serialize_dict, img, genres, category_num, batch_size=512):
        print("validate class init")
        validate_csv = pd.read_csv(validate_csv)
        self.items = list(dict.fromkeys(validate_csv['asin']))
        self.item = set(self.items)
        self.item_user_dict = {}
        # 构建完成 item->user dict
        for it, item_df in validate_csv.groupby('asin'):
            users = item_df['reviewerID']
            users = [user_serialize_dict.get(u) for u in users]
            self.item_user_dict[it] = users
        self.img_dict = img
        self.genres_dict = genres
        self.category_num = category_num
        self.batch_size = batch_size
        item_serialize_dict = {item: item_idx for item_idx, item in enumerate(self.items)}
        self.item_category_tensor, self.item_image_feature_tensor = build_item_feature_tensors(
            item_serialize_dict=item_serialize_dict,
            img_features=self.img_dict,
            genres=self.genres_dict,
            category_num=self.category_num,
        )

    def start_validate(self, model):
        # 开始评估
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        model = model.to(device)
        item_category_tensor = self.item_category_tensor.to(device)
        item_image_feature_tensor = self.item_image_feature_tensor.to(device)
        for batch_start in range(0, len(self.items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.items))
            batch_items = self.items[batch_start:batch_end]
            genres = item_category_tensor[batch_start:batch_end]
            image_feature = item_image_feature_tensor[batch_start:batch_end]
            with torch.no_grad():
                recommend_user_batches = get_similar_user_batch(model, genres, image_feature, max_k)
            for it, recommend_users in zip(batch_items, recommend_user_batches):
                # 计算hr指标
                hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5)
                hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10)
                hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20)
                # 计算NDCG指标
                ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
                ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
                ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)
        item_len = len(self.items)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5/item_len
        ndcg_10 = ndcg_sum_10/item_len
        ndcg_20 = ndcg_sum_20/item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20


if __name__ == '__main__':
    # 参数解析器
    # 参数解析器
    import pickle
    from support import RatingDataset
    from model import CCFCRec
    args = get_args()
    # 提取user的原id: 序列化id的dict
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/test_rating.csv'
    train_df = pd.read_csv(train_path)
    load_dir = 'result/2022-10-14/'
    pkl_file = open(load_dir+'save_dict.pkl', 'rb')
    data = pickle.load(pkl_file)
    dataSet = RatingDataset(train_df, data['img_feature_dict'], data['asin_category_int_map'], data['category_ser_map_len'],
                            data['user_ser_dict'], args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=data['user_ser_dict'], img=data['img_feature_dict'],
                         genres=data['asin_category_int_map'], category_num=data['category_ser_map_len'])
    myModel = CCFCRec(args)
    print('---------数据集加载完毕，开始测试----------------')
    test_result_name = 'test_result.csv'
    with open(test_result_name, 'a+') as f:
        f.write("p@5,p@10,p@20,ndcg@5,ndcg@10,ndcg@20\n")
    load_array = ['98', '99', '100']
    for model in load_array:
        myModel.load_state_dict(torch.load(load_dir+'/'+model+'.pt'))
        hr5, hr_10, hr_20, n_5, n_10, n_20 = validator.start_validate(myModel)
        with open(test_result_name, 'a+') as f:
            f.write("{},{},{},{},{},{}\n".format(p5, p_10, p_20, n_5, n_10, n_20))
