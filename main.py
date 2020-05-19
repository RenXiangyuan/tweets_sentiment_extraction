# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: main.py
@time: 2020-05-18 23:00
"""

import argparse
import os
from model import train, ensemble_infer, TweetModel, TweetDataset, eval_fn
from config import Config
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--cuda_device",
    default='0',
    type=str,
    required=False,
    help="Which GPU To Use",
)

parser.add_argument(
    "--lr",
    default=5,
    type=int,
    required=False,
    help="Learning Rate(1e-5)",
)
parser.add_argument(
    "--bs",
    default=32,
    type=int,
    required=False,
    help="batch size",
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

# args.lr = 5
# args.bs = 32
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# print("Warning, Use Hardcode Setting, not argparser Setting")

config = Config(
    # train_dir='/mfs/renxiangyuan/tweets/data/train_folds.csv',  # 原始数据
    train_dir='/mfs/renxiangyuan/tweets/data/train_folds_extra.csv',  # 加入更多sentimen分类数据

    # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-5-fold-ak',  # 基于ak数据训
    # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-squad-5-fold-ak',  # 基于ak数据训
    model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak',  # 基于ak数据训
    # model_save_dir='/mfs/renxiangyuan/tweets/output/roberta-base-multisent-5-fold-ak',  # 基于ak数据训
    # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-smooth-5-fold-ak',  # 基于ak数据训
    # model_save_dir='/mfs/renxiangyuan/tweets/output/bart-5-fold-ak',  # 基于ak数据训
    # model_save_dir='/mfs/renxiangyuan/tweets/output/test',  # 基于ak数据训

    batch_size=args.bs,
    seed=42,
    lr=args.lr * 1e-5,
    # model_type='bart',
    model_type='roberta',
    do_IO=False, alphe=0.5,
    multi_sent_loss_ratio=0,
    max_seq_length=192,
    num_hidden_layers=13,
    cat_n_layers=2,
    froze_n_layers=0,
)

from utils import set_seed

config.print_info()
set_seed(config.seed)

mode = []
mode.append("train")
# mode.append("test")
# mode.append('evaluate')

# 训练
if "train" in mode:
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    jaccard_scores = []
    for i in range(5):
        jaccard_scores.append(train(fold=i, config=config))
    for i, res_i in enumerate(jaccard_scores):
        print(i, res_i)
    print("mean", np.mean([max(scores) for scores in jaccard_scores]))
    config.print_info()

# 测试
if "test" in mode:
    # model_paths = [
    #     "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_0_epoch_2.pth",
    #     "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_1_epoch_2.pth",
    #     "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_2_epoch_3.pth",
    #     "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_3_epoch_3.pth",
    #     "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_4_epoch_3.pth",
    # ]
    # ensemble_infer(model_paths, config)
    ensemble_infer(model_paths=None, config=config)

# # 评估
if "evaluate" in mode:
    device = torch.device("cuda")
    model = TweetModel(conf=config.model_config, config=config)
    model.to(device)
    res = np.zeros((5, 2))
    for fold in range(5):
        dfx = pd.read_csv(config.TRAINING_FILE)
        df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

        valid_dataset = TweetDataset(
            tweet=df_valid.text.values,
            sentiment=df_valid.sentiment.values,
            selected_text=df_valid.selected_text.values,
            config=config,
        )

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=2
        )

        state_dict_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{fold}_epoch_2.pth")
        print(state_dict_dir)
        model.load_state_dict(torch.load(state_dict_dir))
        model.eval()

        jaccards = eval_fn(valid_data_loader, model, device, config)
        print(jaccards)
        res[fold][0] = jaccards

        state_dict_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{fold}_epoch_3.pth")
        print(state_dict_dir)
        model.load_state_dict(torch.load(state_dict_dir))
        model.eval()

        jaccards = eval_fn(valid_data_loader, model, device, config)
        print(jaccards)
        res[fold][1] = jaccards
    for i, res_i in enumerate(res):
        print(i, res_i[0], res_i[1])
    print("mean", np.mean([max(scores) for scores in res]))