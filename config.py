# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: config.py
@time: 2020-05-16 11:53
"""
import os
import tokenizers, transformers

from utils import SentencePieceTokenizer


class Config(object):
    def __init__(self,
                 train_dir='/mfs/renxiangyuan/tweets/data/train_folds.csv',
                 model_save_dir='.',
                 batch_size=32, seed=42, lr=4e-5, model_type="roberta-base", max_seq_length=128,
                 num_hidden_layers=12, cat_n_layers=1, froze_n_layers=-1,
                 epochs=3,

                 io_loss_ratio=0, io_loss_type='lovasz',
                 smooth=0, mask_pad_loss=False,
                 multi_sent_loss_ratio=0,
                 warmup_samples=0, frozen_warmup=False,
                 warmup_scheduler="linear",
                 fp16=False,
                 clean_data=False,
                 conv_head=False, eps=1e-6, shuffle_seed=-1, init_seed=-1):
        self.TRAINING_FILE = train_dir
        self.MODEL_SAVE_DIR = model_save_dir + f'/{round(lr * 1e5)}e-05lr_{batch_size}bs_{seed}sd'+\
                              f'_{max_seq_length}len_{num_hidden_layers}layer_{cat_n_layers}cat_{froze_n_layers}froze'
        self.TRAIN_BATCH_SIZE = batch_size
        self.seed = seed
        self.lr = lr
        self.model_type = model_type
        self.MAX_LEN = max_seq_length
        self.num_hidden_layers = num_hidden_layers
        self.cat_n_layers = cat_n_layers
        self.froze_n_layers = froze_n_layers

        self.EPOCHS = epochs
        if self.EPOCHS != 3:
            self.MODEL_SAVE_DIR += f"_{self.EPOCHS}ep"

        self.conv_head = conv_head
        if self.conv_head: self.MODEL_SAVE_DIR += f"_conv"
        self.smooth = smooth
        if smooth > 0:
            self.MODEL_SAVE_DIR += f"_{smooth}smooth"
        self.multi_sent_loss_ratio = multi_sent_loss_ratio
        if multi_sent_loss_ratio > 0:
            self.MODEL_SAVE_DIR += f"_{multi_sent_loss_ratio}beta"
        self.multi_sent_class = {'anger': 0, 'boredom': 1, 'empty': 2, 'enthusiasm': 3, 'fun': 4, 'happiness': 5,
                                 'hate': 6, 'love': 7, 'neutral': 8, 'relief': 9, 'sadness': 10, 'surprise': 11,
                                 'worry': 12}

        self.io_loss_ratio = io_loss_ratio
        self.io_loss_type = io_loss_type
        if self.io_loss_ratio > 0:
            self.MODEL_SAVE_DIR += f"_{io_loss_ratio}{io_loss_type}"

        self.warmup_iters = warmup_samples//batch_size
        self.frozen_warmup = frozen_warmup
        if self.warmup_iters > 0:
            self.MODEL_SAVE_DIR += f"_{warmup_samples}warm"
        if frozen_warmup:
            assert self.warmup_iters >= 0
            self.MODEL_SAVE_DIR += f"_fwarm"

        self.warmup_scheduler = warmup_scheduler
        if warmup_scheduler == 'cosine':
            self.MODEL_SAVE_DIR += '_cos'

        self.mask_pad_loss = mask_pad_loss
        if mask_pad_loss:
            self.MODEL_SAVE_DIR += '_maskloss'

        self.fp16 = fp16
        if fp16:
            self.MODEL_SAVE_DIR += f"_fp16"

        self.clean_data = clean_data
        if clean_data:
            assert "clean-data" in self.MODEL_SAVE_DIR

        self.eps = eps
        if eps != 1e-6:
            self.MODEL_SAVE_DIR += f'_{eps}adameps'
        self.shuffle_seed = shuffle_seed
        if shuffle_seed != -1:
            self.MODEL_SAVE_DIR += f'_{shuffle_seed}shufflesd'
        self.init_seed = init_seed
        if init_seed != -1:
            self.MODEL_SAVE_DIR += f"_{init_seed}initsd"

        self.ACCUMULATION_STEPS = 1
        self.MAX_GRAD_NORM = 1.0
        self.n_worker_train = 16
        self.VALID_BATCH_SIZE = 32

        if 'roberta' in self.model_type:
            if self.model_type == 'roberta-squad':
                self.ROBERTA_PATH = "/mfs/pretrain/roberta-base-squad2"; print("Using Squad Roberta")  # Roberta Squad 2
            elif self.model_type == 'roberta-base':
                self.ROBERTA_PATH = "/mfs/pretrain/roberta-base"; assert 'squad' not in model_save_dir
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
            self.model_config.max_length = self.MAX_LEN
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
        elif 'albert' in self.model_type:
            assert self.MAX_LEN >= 132, "Albert MaxLen至少132"
            if self.model_type == "albert-base":
                self.ALBERT_PATH = "/mfs/pretrain/albert-base-v2"
            elif self.model_type == "albert-large":
                self.ALBERT_PATH ="/mfs/pretrain/albert-large-v2"
            elif self.model_type == "albert-xlarge":
                self.ALBERT_PATH ="/mfs/pretrain/albert-xlarge-v2"
            else:
                raise NotImplementedError()
            self.TOKENIZER = SentencePieceTokenizer(self.ALBERT_PATH)
            self.model_config = transformers.AlbertConfig.from_pretrained(self.ALBERT_PATH)
            self.model_config.output_hidden_states = True
            self.model_config.max_length = max_seq_length
        else:
            raise NotImplementedError()

    def print_info(self):
        print("Config Info")
        print(f"\tdevice:              {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"\tSeed:                {self.seed}")
        print(f"\tLearning Rate:       {self.lr}")
        print(f"\tBatch Size:          {self.TRAIN_BATCH_SIZE}")
        print(f"\tMax Length:          {self.MAX_LEN}")
        print(f"\tmodel_type:          {self.model_type}")
        print(f"\tmodel_save_dir:      {self.MODEL_SAVE_DIR}")
        print(f"\ttrain_dir:           {self.TRAINING_FILE}")
        print(f"\tInit N Hidden Layer: {self.num_hidden_layers}")
        print(f"\tCat N Hidden:        {self.cat_n_layers}")
        print(f"\tFreeze N Hidden:     {self.froze_n_layers}")


