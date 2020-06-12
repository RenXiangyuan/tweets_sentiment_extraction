# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: main.py
@time: 2020-05-18 23:00
"""

# nohup python main.py 1>.log 2>&1 &

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from model import train, ensemble_infer, TweetModel, TweetDataset, eval_fn
from utils import set_seed


def main(args, mode):
    config = Config(
        train_dir='/mfs/renxiangyuan/tweets/data/train_folds.csv',  # 原始数据
        # train_dir='/mfs/renxiangyuan/tweets/data/train_folds_extra.csv',  # 加入更多sentimen分类数据

        model_save_dir=f'/mfs/renxiangyuan/tweets/output/{args.model_type}-5-fold-ak',
        # model_save_dir=f'/mfs/renxiangyuan/tweets/output/shuffle/{args.model_type}-5-fold-ak',

        model_type=args.model_type,
        batch_size=args.bs,
        seed=args.seed,
        lr=args.lr * 1e-5,
        max_seq_length=args.max_seq_length,
        num_hidden_layers=args.num_hidden_layers,
        cat_n_layers=args.cat_n_layers,
        froze_n_layers=args.froze_n_layers,

        # conv_head=True,
        # eps=args.eps,
        shuffle_seed=args.shuffle_seed,
        init_seed=args.init_seed,

        epochs=args.epochs,  # 默认epochs=3
        warmup_samples=args.warmup_samples,
        # frozen_warmup=False,
        warmup_scheduler=args.scheduler,
        mask_pad_loss=args.mask_pad_loss,
        smooth=args.smooth,
        # fp16=False,

        io_loss_ratio=args.io_loss_ratio, io_loss_type=args.io_loss_type,
        # multi_sent_loss_ratio=0,
        # clean_data=True,  # 模型clean_data=False
    )

    config.print_info()

    set_seed(config.seed)

    # 训练
    if "train" in mode:
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        jaccard_scores = []
        for i in args.train_folds:
            scores_i = train(fold=i, config=config)
            jaccard_scores.append(scores_i)
            # if i == 0 and max(scores_i) < 0.705:
            #     print("Fold 0 Too Weak, Early Stop")
            #     break
        for i, res_i in enumerate(jaccard_scores):
            print(i, res_i)
        print("mean", np.mean([max(scores) for scores in jaccard_scores]))
        for i in range(1, config.EPOCHS):
            print(f"\tEpoch{i+1}: ", np.mean([scores[i] for scores in jaccard_scores]))
        config.print_info()

    # 测试
    if "test" in mode:
        model_paths = [
            "/mfs/renxiangyuan/tweets/output/shuffle/roberta-squad-5-fold-ak/4e-05lr_32bs_42sd_128len_12layer_1cat_-1froze_11shufflesd/model_0_epoch_2.pth",
            "/mfs/renxiangyuan/tweets/output/shuffle/roberta-squad-5-fold-ak/4e-05lr_32bs_42sd_128len_12layer_1cat_-1froze_3shufflesd/model_1_epoch_3.pth",
            "/mfs/renxiangyuan/tweets/output/shuffle/roberta-squad-5-fold-ak/4e-05lr_32bs_42sd_128len_12layer_1cat_-1froze_18shufflesd/model_2_epoch_3.pth",
            "/mfs/renxiangyuan/tweets/output/shuffle/roberta-squad-5-fold-ak/4e-05lr_32bs_42sd_128len_12layer_1cat_-1froze_13shufflesd/model_3_epoch_2.pth",
            "/mfs/renxiangyuan/tweets/output/shuffle/roberta-squad-5-fold-ak/4e-05lr_32bs_42sd_128len_12layer_1cat_-1froze_19shufflesd/model_4_epoch_3.pth",
            # "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_0_epoch_2.pth",
            # "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_1_epoch_2.pth",
            # "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_2_epoch_3.pth",
            # "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_3_epoch_3.pth",
            # "/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak/5e-05lr_32bs_42sd_13layer/model_4_epoch_3.pth",
        ]
        ensemble_infer(model_paths, config)
        # ensemble_infer(model_paths=None, config=config)

    # # 评估
    if "evaluate" in mode:
        device = torch.device("cuda")
        model = TweetModel(conf=config.model_config, config=config)
        model.to(device)
        res = [[] for _ in range(5)]
        for fold in range(5):
            dfx = pd.read_csv(config.TRAINING_FILE)
            df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

            valid_dataset = TweetDataset(
                tweet=df_valid.text.values,
                sentiment=df_valid.sentiment.values,
                selected_text=df_valid.selected_text.values,
                config=config,
            )

            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size=config.VALID_BATCH_SIZE,
                num_workers=8
            )

            for ep in range(1, config.EPOCHS):
                state_dict_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{fold}_epoch_{ep+1}.pth")
                print(state_dict_dir)
                model.load_state_dict(torch.load(state_dict_dir))
                model.eval()

                jaccards = eval_fn(valid_data_loader, model, device, config)
                print(jaccards)
                res[fold].append(jaccards)

        for i, res_i in enumerate(res):
            print(i, res_i)
        print("mean", np.mean([max(scores) for scores in res]))

        for i in range(2):
            print(f"\tEpoch{i + 1}: ", np.mean([scores[i] for scores in res]))


if __name__ == "__main__":
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
        "--model_type",
        default='roberta-base',  # albert-large, albert-base, roberta-squad, bart
        type=str,
        required=False,
        help="Which Pretrain Model Use",
    )

    parser.add_argument(
        "--lr",
        default=4,
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

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help="Random Seed",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        required=False,
        help="Max Sequence Length",
    )

    parser.add_argument(
        "--cat_n_layers",
        default=1,
        type=int,
        required=False,
        help="Cat Last N Layers Hidden State",
    )

    parser.add_argument(
        "--froze_n_layers",
        default=-1,
        type=int,
        required=False,
        help="Froze Which Layer During Training",
    )

    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        required=False,
        help="Number of Hidden Layers To Init (currently only for roberta)",
    )

    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        required=False,
        help="Number of Epochs during Training",
    )

    parser.add_argument(
        "--smooth",
        default=0,
        type=float,
        required=False,
        help="Label Smoothing Alpha",
    )

    parser.add_argument(
        "--io_loss_ratio",
        default=0,
        type=float,
        required=False,
        help="IO Loss Alpha",
    )

    parser.add_argument(
        "--io_loss_type",
        default="lovasz",
        type=str,
        required=False,
        help="IO Loss Function",
    )

    parser.add_argument(
        "--eps",
        default=1e-6,
        type=float,
        required=False,
        help="AdamW Epsilon",
    )

    args = parser.parse_args()

    args.warmup_samples = 0
    args.frozen_warmup=False
    args.scheduler = 'linear'
    args.mask_pad_loss = False
    args.shuffle_seed = -1
    args.init_seed = -1
    args.train_folds = [0, 1, 2, 3, 4]



    args.model_type = "roberta-squad"
    # args.lr = 4
    # args.bs = 32
    # args.num_hidden_layers = 13
    # args.froze_n_layers = 0
    # args.cat_n_layers = 2
    # args.epochs = 3
    # args.smooth = 0
    args.cuda_device = '9'
    args.io_loss_ratio = 0.4
    print("Warning, Use Hardcode Setting, not argparser Setting")

    mode = [
       #  "train",
        # "test"
        "evaluate",
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    main(args, mode)

    # import copy

    # args1 = copy.deepcopy(args)
    # args1.model_type = "roberta-base"
    # # args1.smooth = 0.2
    # # args1.mask_pad_loss = True
    # # args1.io_loss_ratio = 0.5
    # args1.epochs=4
    # main(args1, mode)