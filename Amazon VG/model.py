import math
import os
import sys
import pickle
import random
import functools
import json
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle
from support import RatingDataset
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring


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


def set_random_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_category_reweight(item_genres, args):
    if args.method_variant != 'weak_q_reweight':
        return None
    if args.weak_loss_alpha <= 0:
        return None
    category_count = (item_genres != -1).sum(dim=1).float()
    weights = torch.ones_like(category_count)
    weights = torch.where(category_count <= args.weak_cat_threshold, weights + args.weak_loss_alpha, weights)
    return weights / weights.mean().detach()


def build_adaptive_qbpr_weights(item_genres, support_confidence, args):
    if getattr(args, "method_variant", "baseline") != "adaptive_conf_qbpr":
        return None
    if getattr(args, "adaptive_loss_alpha", 0) <= 0:
        return None
    category_count = (item_genres != -1).sum(dim=1).float()
    weak_mask = category_count <= args.weak_cat_threshold
    support_confidence = torch.clamp(support_confidence.float().to(category_count.device), min=0.0, max=1.0)
    weights = torch.ones_like(category_count)
    weights = weights + args.adaptive_loss_alpha * weak_mask.float() * support_confidence
    return weights / weights.mean().detach()


def weighted_sum(per_item_loss, weights, enabled):
    if weights is None or enabled is False:
        return per_item_loss.sum()
    return (per_item_loss * weights).sum()


def validate_method_args(args):
    method_variant = getattr(args, "method_variant", "baseline")
    reweight_flags = [
        getattr(args, "reweight_q_bpr", False),
        getattr(args, "reweight_self_contrast", False),
        getattr(args, "reweight_contrast", False),
    ]
    if method_variant != "weak_q_reweight" and any(reweight_flags):
        raise ValueError("reweight flags can only be used with method_variant=weak_q_reweight")
    if method_variant == "category_conf_input":
        if int(getattr(args, "category_conf_dim", 16)) <= 0:
            raise ValueError("category_conf_dim must be positive for category_conf_input")
        if int(getattr(args, "category_conf_max_count", 5)) <= 0:
            raise ValueError("category_conf_max_count must be positive for category_conf_input")
    if method_variant == "adaptive_conf_qbpr":
        if float(getattr(args, "adaptive_loss_alpha", 1.0)) <= 0:
            raise ValueError("adaptive_loss_alpha must be positive for adaptive_conf_qbpr")
        if int(getattr(args, "adaptive_history_max_count", 20)) <= 0:
            raise ValueError("adaptive_history_max_count must be positive for adaptive_conf_qbpr")


def scalar_text(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def training_result_header():
    return (
        "checkpoint_index,epoch,batch,total_batches,elapsed_s,"
        "loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n"
    )


def build_run_config(args, model):
    method_variant = getattr(args, "method_variant", "baseline")
    return {
        "method_variant": method_variant,
        "category_conf_dim": int(getattr(args, "category_conf_dim", 0)),
        "category_conf_max_count": int(getattr(args, "category_conf_max_count", 0)),
        "category_bin_count": int(getattr(model, "category_bin_count", 0)),
        "gen_layer1_input_dim": int(model.gen_layer1.in_features),
        "gen_layer1_output_dim": int(model.gen_layer1.out_features),
        "weak_cat_threshold": int(getattr(args, "weak_cat_threshold", 3)),
        "weak_loss_alpha": float(getattr(args, "weak_loss_alpha", 0.0)),
        "adaptive_loss_alpha": float(getattr(args, "adaptive_loss_alpha", 0.0)),
        "adaptive_history_max_count": int(getattr(args, "adaptive_history_max_count", 0)),
        "seed": int(getattr(args, "seed", -1)),
        "num_workers": int(getattr(args, "num_workers", 0)),
    }


def write_run_config(args, model, model_save_dir):
    with open(os.path.join(model_save_dir, "run_config.json"), "w") as f:
        json.dump(build_run_config(args, model), f, indent=2, sort_keys=True)


def build_training_result_row(
    checkpoint_index,
    epoch,
    batch,
    total_batches,
    elapsed_s,
    loss,
    contrast_sum,
    metrics,
):
    values = [
        checkpoint_index,
        epoch,
        batch,
        total_batches,
        elapsed_s,
        scalar_text(loss),
        scalar_text(contrast_sum),
        *[scalar_text(metric) for metric in metrics],
    ]
    return ",".join(str(value) for value in values) + "\n"


def format_training_progress(epoch, total_epochs, batch, total_batches, checkpoint_index, total_loss, elapsed_s):
    return (
        f"[epoch {epoch}/{total_epochs}]"
        f"[batch {batch}/{total_batches}]"
        f"[ckpt {checkpoint_index}] "
        f"total_loss:{scalar_text(total_loss)}, elapsed:{elapsed_s}s"
    )


# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()
        self.args = args
        self.method_variant = getattr(args, "method_variant", "baseline")
        self.category_conf_dim = int(getattr(args, "category_conf_dim", 16))
        self.category_conf_max_count = int(getattr(args, "category_conf_max_count", 5))
        self.category_bin_count = 4
        if self.uses_category_confidence():
            if self.category_conf_dim <= 0:
                raise ValueError("category_conf_dim must be positive for category_conf_input")
            if self.category_conf_max_count <= 0:
                raise ValueError("category_conf_max_count must be positive for category_conf_input")
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))
        # 定义属性attribute注意力层
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        # 控制整个模型的激活函数
        self.h = nn.LeakyReLU()
        # 图像的映射矩阵
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()  # 将门控信号映射到[0, 1]之间
        # user和item的嵌入层，可用预训练的进行初始化
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
        # 定义生成层，将(q_v_a, u)的信息，共同生成 q_v_c， 生成包含协同信息的item嵌入
        if self.uses_category_confidence():
            self.category_conf_embedding = nn.Embedding(self.category_bin_count, self.category_conf_dim)
        gen_input_dim = args.attr_present_dim + args.implicit_dim + self.category_conf_extra_dim()
        self.gen_layer1 = nn.Linear(gen_input_dim, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # 参数初始化
        self.__init_param__()

    def uses_category_confidence(self):
        return self.method_variant == "category_conf_input"

    def category_conf_extra_dim(self):
        if self.uses_category_confidence():
            return self.category_conf_dim + 2
        return 0

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.image_projection)
        # 生成层初始化
        # user, item嵌入层的初始化, 没有预训练的情况下就初始化
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)
        if self.uses_category_confidence():
            nn.init.xavier_normal_(self.category_conf_embedding.weight)

    def build_category_conf_bins(self, attribute):
        category_count = (attribute != -1).sum(dim=1)
        bins = torch.zeros_like(category_count, dtype=torch.long)
        bins = torch.where((category_count > 0) & (category_count <= 3), torch.ones_like(bins), bins)
        bins = torch.where(category_count == 4, torch.full_like(bins, 2), bins)
        bins = torch.where(category_count >= 5, torch.full_like(bins, 3), bins)
        return bins

    def build_category_conf_features(self, attribute):
        category_count = (attribute != -1).sum(dim=1).float()
        count_clamped = torch.clamp(category_count, min=0, max=self.category_conf_max_count)
        category_density = count_clamped / float(self.category_conf_max_count)
        category_log_norm = torch.log1p(count_clamped) / math.log1p(float(self.category_conf_max_count))
        scalar_features = torch.stack((category_log_norm, category_density), dim=1)
        category_bin = self.build_category_conf_bins(attribute)
        category_conf_emb = self.category_conf_embedding(category_bin)
        return torch.cat((category_conf_emb, scalar_features.to(category_conf_emb.dtype)), dim=1)

    def build_generator_input(self, final_attr_emb, p_v, attribute):
        if not self.uses_category_confidence():
            return torch.cat((final_attr_emb, p_v), dim=1)
        category_conf_features = self.build_category_conf_features(attribute)
        return torch.cat((final_attr_emb, p_v, category_conf_features), dim=1)

    def encode_content_components(self, attribute, image_feature, batch_size):
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(device)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        image_norm = torch.nn.functional.normalize(image_feature, dim=1)
        p_v = torch.matmul(image_feature, self.image_projection)  # item的图像嵌入向量
        q_v_a = self.build_generator_input(final_attr_emb, p_v, attribute)
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c, final_attr_emb, p_v

    def forward(self, attribute, image_feature, batch_size):
        q_v_c, _, _ = self.encode_content_components(attribute, image_feature, batch_size)
        return q_v_c


def train(model, train_loader, optimizer, valida, args, model_save_dir):
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 写入超参数
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_run_config(args, model, model_save_dir)
    with open(test_save_path, 'a+') as f:
        f.write(training_result_header())
    save_index = 0
    total_batches = len(train_loader)
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        for user, item, item_genres, item_img_feature, neg_user, positive_item_list, negative_item_list, self_neg_list, support_confidence in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)
            support_confidence = support_confidence.to(device)
            # run model
            q_v_c = model(item_genres, item_img_feature, user.shape[0])
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
            # compute contrast loss
            positive_item_emb = model.item_embedding[positive_item_list]
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10
            # negative samples
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_per_item = torch.sum(contrast_val, dim=1) / contrast_val.shape[1]
            category_weights = build_category_reweight(item_genres, args)
            contrast_sum = weighted_sum(contrast_per_item, category_weights, args.reweight_contrast)
            '''
            contrast self
            '''
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2)/(
                args.tau*torch.norm(q_v_c_unsqueeze, dim=2)*torch.norm(self_neg_item_emb, dim=2))
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                    args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(self_pos_contrast_exp/(self_pos_contrast_exp+self_neg_contrast_sum))
            self_contrast_sum = weighted_sum(self_contrast_val, category_weights, args.reweight_self_contrast)
            # rank loss
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).sum()
            # 使用属性生成item嵌入，再做一个bpr排序
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_ukv2_per_item = -logsigmoid(y_uv2 - y_kv2)
            adaptive_qbpr_weights = build_adaptive_qbpr_weights(item_genres, support_confidence, args)
            if adaptive_qbpr_weights is not None:
                y_ukv2 = weighted_sum(y_ukv2_per_item, adaptive_qbpr_weights, True)
            else:
                y_ukv2 = weighted_sum(y_ukv2_per_item, category_weights, args.reweight_q_bpr)
            total_loss = args.lambda1*(contrast_sum+self_contrast_sum) + (1-args.lambda1)*(y_ukv+y_ukv2)
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            if i_batch % args.save_batch_time == 0:
                model.eval()
                elapsed_s = int(time.time()-batch_time)
                checkpoint_index = save_index + 1
                print(format_training_progress(
                    i_epoch + 1,
                    args.epoch,
                    i_batch,
                    total_batches,
                    checkpoint_index,
                    total_loss,
                    elapsed_s,
                ))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                with open(test_save_path, 'a+') as f:
                    f.write(build_training_result_row(
                        checkpoint_index,
                        i_epoch + 1,
                        i_batch,
                        total_batches,
                        elapsed_s,
                        total_loss,
                        contrast_sum,
                        (hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20),
                    ))
                # 保存模型
                batch_time = time.time()
                save_index = checkpoint_index
                torch.save(model.state_dict(), model_save_dir + '/' + str(save_index)+".pt")


if __name__ == '__main__':
    # args
    args = get_args()
    validate_method_args(args)
    # result save dir
    save_dir = os.path.join(args.result_root, time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())))
    os.makedirs(save_dir, exist_ok=False)
    set_random_seed(args.seed)
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/validate_rating.csv'
    train_df = pd.read_csv(train_path)
    total_user_set = train_df['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    # write internal variable
    with open(save_dir+"/save_dict.pkl", "wb") as file:
        save_dict = {'img_feature_dict': img_feature_dict, 'asin_category_int_map': asin_category_int_map,
                     'category_ser_map_len': category_ser_map.__len__(), 'user_ser_dict': user_ser_dict}
        pickle.dump(save_dict, file)
    # load dataset
    dataSet = RatingDataset(train_df, img_feature_dict, asin_category_int_map, category_ser_map.__len__(),
                            user_ser_dict, args.positive_number, args.negative_number,
                            adaptive_history_max_count=args.adaptive_history_max_count)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
    }
    if args.pin_memory:
        loader_kwargs['pin_memory'] = True
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        if args.multiprocessing_context:
            loader_kwargs['multiprocessing_context'] = args.multiprocessing_context
    if args.seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader_kwargs['generator'] = generator
        loader_kwargs['worker_init_fn'] = functools.partial(seed_worker, base_seed=args.seed)
    train_loader = torch.utils.data.DataLoader(dataSet, **loader_kwargs)
    print("模型超参数:", args_tostring(args))
    myModel = CCFCRec(args)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=user_ser_dict, img=img_feature_dict,
                         genres=asin_category_int_map, category_num=category_ser_map.__len__())
    train(myModel, train_loader, optimizer, validator, args, save_dir)
