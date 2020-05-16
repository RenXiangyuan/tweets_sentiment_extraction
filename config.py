# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: config.py
@time: 2020-05-16 11:53
"""

import tokenizers, transformers


class Config(object):
    def __init__(self, train_dir, model_save_dir, batch_size=128, seed=42, lr=3e-5, model_type='roberta', alphe=0.3,
                 do_IO=False, smooth=0, multi_sent=False):
        self.seed = seed
        self.lr = lr
        self.model_type = model_type
        self.TRAINING_FILE = train_dir  # '/data/nfs/fangzhiqiang/nlp_data/tweet_extraction/folds/train_folds.csv'# ak数据
        self.alpha = alphe
        self.do_IO = do_IO
        self.TRAIN_BATCH_SIZE = batch_size  # 16
        self.MODEL_SAVE_DIR = model_save_dir + f'/{round(self.lr * 1e5)}e-05lr_{batch_size}bs_{self.seed}sd'
        self.smooth = smooth
        if smooth > 0:
            self.MODEL_SAVE_DIR += f"_{smooth}ls"
        if self.do_IO:
            self.MODEL_SAVE_DIR += f"_{alphe}alpha"
        self.multi_sent = multi_sent
        self.alpha_multi_sent = 0.3
        self.multi_sent_class = {'anger': 0, 'boredom': 1, 'empty': 2, 'enthusiasm': 3, 'fun': 4, 'happiness': 5,
                                 'hate': 6, 'love': 7, 'neutral': 8, 'relief': 9, 'sadness': 10, 'surprise': 11,
                                 'worry': 12}
        self.MAX_LEN = 192
        # self.loss_type = 'bce'
        self.loss_type = 'lovasz'
        self.eps = 1e-6
        self.ACCUMULATION_STEPS = 1
        self.VALID_BATCH_SIZE = 16
        self.EPOCHS = 3
        self.MAX_GRAD_NORM = 1.0

        if self.model_type == 'roberta':
            self.ROBERTA_PATH = "/mfs/fangzhiqiang/nlp_model/roberta-base"
            # self.ROBERTA_PATH = "/mfs/pretrain/roberta-base-squad2"; print("Using Squad Roberta")  # Roberta Squad 2
            self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                vocab_file=f"{self.ROBERTA_PATH}/vocab.json",
                merges_file=f"{self.ROBERTA_PATH}/merges.txt",
                lowercase=True,
                add_prefix_space=True
            )
            self.model_config = transformers.BertConfig.from_pretrained(self.ROBERTA_PATH)
            self.model_config.output_hidden_states = True
        elif self.model_type == 'electra':
            self.ELECTRA_PATH = "/mfs/fangzhiqiang/nlp_model/electra-base-discriminator-2/"
            self.TOKENIZER = tokenizers.BertWordPieceTokenizer(f"{self.ELECTRA_PATH}/vocab.txt", lowercase=True)
            self.model_config = transformers.ElectraConfig.from_pretrained(self.ELECTRA_PATH)
            self.model_config.output_hidden_states = True

    def print(self):
        print(f"Seed\t: {self.seed}")
        print(f"Learning Rate\t: {self.lr}")
        print(f"Batch Size\t: {self.batch_size}")
        print(f"Alpha\t: {self.alpha}")
        print(f"model: {self.model_type}")
        print(f"model_save_dir: {self.MODEL_SAVE_DIR}")
        print(f"train_dir: {self.TRAINING_FILE}")

config = Config(
        # train_dir='/mfs/renxiangyuan/tweets/data/train_folds.csv',  # 原始数据
        train_dir='/mfs/renxiangyuan/tweets/data/train_folds_extra.csv',  # 加入更多sentimen分类数据

        # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-5-fold-ak',  # 基于ak数据训
        # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-sqauad-5-fold-ak',  # 基于ak数据训
        # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-5-fold-ak',  # 基于ak数据训
        model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multisent-5-fold-ak',  # 基于ak数据训
        # model_save_dir = '/mfs/renxiangyuan/tweets/output/test',  # 基于ak数据训

        batch_size = 32,
        # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-smooth-5-fold-ak',  # 基于ak数据训
        seed=42,
        lr=3e-5,
        # model_type='electra'
        model_type='roberta',
        alphe=0.5,
        do_IO=False,
        multi_sent = True,
    )