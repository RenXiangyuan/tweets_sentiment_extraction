# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: main.py
@time: 2020-06-10 11:33
"""
import tokenizers
import pandas as pd
import transformers
import os
import numpy as np
from sentiment_pretrain.train_eval import train
from utils import set_seed


class Config(object):
    def __init__(self,
                 MODEL_SAVE_DIR,
                 model_type = "roberta-squad",
                 num_hidden_layers = 12, max_length = 128, TRAIN_BATCH_SIZE=32,
                 shuffle_seed=-1, seed=42):
        set_seed(seed)

        self.MAX_LEN = max_length

        self.MODEL_SAVE_DIR = MODEL_SAVE_DIR
        self.model_type = model_type

        if 'roberta' in model_type:
            if model_type == 'roberta-squad':
                self.ROBERTA_PATH = "/mfs/pretrain/roberta-base-squad2"; print("Using Squad Roberta")  # Roberta Squad 2
            # elif self.model_type == 'roberta-base':
            #     self.ROBERTA_PATH = "/mfs/pretrain/roberta-base"; assert 'squad' not in model_save_dir
            else:
                raise NotImplementedError

            self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                vocab_file=f"{self.ROBERTA_PATH}/vocab.json",
                merges_file=f"{self.ROBERTA_PATH}/merges.txt",
                lowercase=True,
                add_prefix_space=True
            )
            self.model_config = transformers.BertConfig.from_pretrained(self.ROBERTA_PATH)
            self.model_config.output_hidden_states = True
            self.model_config.max_length = max_length
            self.model_config.num_hidden_layers = num_hidden_layers

        self.lr = 4e-5
        self.eps = 1e-8
        self.warmup_scheduler = 'linear'
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.VALID_BATCH_SIZE = 16
        self.frozen_warmup = False
        self.warmup_iters = 0
        self.froze_n_layers = -1
        self.n_worker_train = 16
        self.ACCUMULATION_STEPS = 1
        self.EPOCHS = 3
        self.shuffle_seed = shuffle_seed
        self.multi_sent_loss_ratio = 0
        if self.multi_sent_loss_ratio > 0:
            self.MODEL_SAVE_DIR += f"_{self.multi_sent_loss_ratio}beta"
        self.multi_sent_class = {'anger': 0, 'boredom': 1, 'empty': 2, 'enthusiasm': 3, 'fun': 4, 'happiness': 5,
                                 'hate': 6, 'love': 7, 'neutral': 8, 'relief': 9, 'sadness': 10, 'surprise': 11,
                                 'worry': 12}

def main():
    config = Config(MODEL_SAVE_DIR = f'/mfs/renxiangyuan/tweets/output/sentiment_pretrain/roberta-squad-5-fold-ak')

    train_csv = pd.read_csv('/mfs/renxiangyuan/tweets/data/train_folds.csv')
    valid_csv = pd.read_csv('/mfs/renxiangyuan/tweets/data/test.csv')

    # train_csv = pd.read_csv('/mfs/renxiangyuan/tweets/data/train_folds_extra.csv')
    # valid_csv = pd.read_csv('')

    train_np = np.array(train_csv)
    valid_np = np.array(valid_csv)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    scores = train(train_np, valid_np, config=config)
    print(scores)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()