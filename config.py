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
                 do_IO=False, smooth=0, multi_sent_loss_ratio=0.1, max_seq_length=192, num_hidden_layers=12,
                 cat_n_layers=2, froze_n_layers=-1):
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
        self.multi_sent_loss_ratio = multi_sent_loss_ratio
        if multi_sent_loss_ratio > 0:
            self.MODEL_SAVE_DIR += f"_{multi_sent_loss_ratio}beta"
        self.multi_sent_class = {'anger': 0, 'boredom': 1, 'empty': 2, 'enthusiasm': 3, 'fun': 4, 'happiness': 5,
                                 'hate': 6, 'love': 7, 'neutral': 8, 'relief': 9, 'sadness': 10, 'surprise': 11,
                                 'worry': 12}
        self.MAX_LEN = max_seq_length
        if max_seq_length != 192:
            self.MODEL_SAVE_DIR += f"_{max_seq_length}len"
        if num_hidden_layers != 12:
            self.MODEL_SAVE_DIR += f"_{num_hidden_layers}layer"
        # self.loss_type = 'bce'
        self.loss_type = 'lovasz'
        self.eps = 1e-6
        self.ACCUMULATION_STEPS = 1
        self.VALID_BATCH_SIZE = 16
        self.EPOCHS = 3
        self.MAX_GRAD_NORM = 1.0
        self.n_worker_train = 16
        self.cat_n_layers = cat_n_layers
        if self.cat_n_layers == 3:
            self.MODEL_SAVE_DIR += f"_{self.cat_n_layers}cat"
        self.froze_n_layers = froze_n_layers
        if froze_n_layers >= 0:
            self.MODEL_SAVE_DIR += f"_{froze_n_layers}froze"

        if self.model_type == 'roberta':
            if 'roberta-squad' in model_save_dir:
                self.ROBERTA_PATH = "/mfs/pretrain/roberta-base-squad2"; print("Using Squad Roberta")  # Roberta Squad 2
            elif 'roberta-base' in model_save_dir:
                self.ROBERTA_PATH = "/mfs/fangzhiqiang/nlp_model/roberta-base"; assert 'squad' not in model_save_dir
            else:
                raise ValueError("pretrain path与model_save_dir不一致")
            # self.ROBERTA_PATH = "/mfs/pretrain/roberta-base-squad2"; print("Using Squad Roberta")  # Roberta Squad 2

            self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                vocab_file=f"{self.ROBERTA_PATH}/vocab.json",
                merges_file=f"{self.ROBERTA_PATH}/merges.txt",
                lowercase=True,
                add_prefix_space=True
            )
            self.model_config = transformers.BertConfig.from_pretrained(self.ROBERTA_PATH)
            self.model_config.output_hidden_states = True
            # self.model_config.max_length = self.MAX_LEN
            self.model_config.num_hidden_layers = num_hidden_layers
        elif self.model_type == 'electra':
            self.ELECTRA_PATH = "/mfs/fangzhiqiang/nlp_model/electra-base-discriminator-2/"
            self.TOKENIZER = tokenizers.BertWordPieceTokenizer(f"{self.ELECTRA_PATH}/vocab.txt", lowercase=True)
            self.model_config = transformers.ElectraConfig.from_pretrained(self.ELECTRA_PATH)
            self.model_config.output_hidden_states = True
        elif self.model_type == 'bart':
            self.BART_PATH = "/mfs/pretrain/bart-large"
            self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                vocab_file=f"/mfs/fangzhiqiang/nlp_model/roberta-base/vocab.json",
                merges_file=f"/mfs/fangzhiqiang/nlp_model/roberta-base/merges.txt",
                lowercase=True,
                add_prefix_space=True
            )
            # self.TOKENIZER = transformers.BartTokenizer.from_pretrained('/mfs/pretrain/bart-large')
            self.model_config = transformers.BartConfig.from_pretrained(self.BART_PATH)
            self.model_config.output_hidden_states = True
            # self.model_config.max_length = self.MAX_LEN
            self.model_config.num_hidden_layers = num_hidden_layers

    def print_info(self):
        print(f"Seed\t: {self.seed}")
        print(f"Learning Rate\t: {self.lr}")
        print(f"Batch Size:\t {self.TRAIN_BATCH_SIZE}")
        print(f"Max Length:\t{self.MAX_LEN}")
        print(f"Alpha\t: {self.alpha}")
        print(f"model: {self.model_type}")
        print(f"model_save_dir: {self.MODEL_SAVE_DIR}")
        print(f"train_dir: {self.TRAINING_FILE}")
