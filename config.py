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
    def __init__(self, train_dir, model_save_dir, batch_size=128, seed=42, lr=3e-5, model_type='roberta', alphe=0.3,
                 do_IO=False, smooth=0, multi_sent_loss_ratio=0.1, max_seq_length=192, num_hidden_layers=12,
                 cat_n_layers=2, froze_n_layers=-1, warmup_samples=0, frozen_warmup=False, warmup_scheduler="linear",
                 fp16=False, epochs=3, loss_type='lovasz', mask_pad_loss=False):

        self.seed = seed
        self.lr = lr
        self.model_type = model_type
        self.TRAINING_FILE = train_dir  # '/data/nfs/fangzhiqiang/nlp_data/tweet_extraction/folds/train_folds.csv'# ak数据

        self.TRAIN_BATCH_SIZE = batch_size  # 16
        self.MODEL_SAVE_DIR = model_save_dir + f'/{round(self.lr * 1e5)}e-05lr_{batch_size}bs_{self.seed}sd'
        self.smooth = smooth
        if smooth > 0:
            self.MODEL_SAVE_DIR += f"_{smooth}smooth"
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

        self.do_IO = do_IO
        self.alpha = alphe
        self.loss_type = loss_type
        if self.do_IO:
            assert alphe > 0
            self.MODEL_SAVE_DIR += f"_{alphe}alpha"
            if loss_type != 'lovasz':  # 'bce'
                self.MODEL_SAVE_DIR += f"_{loss_type}"
        self.eps = 1e-6
        self.ACCUMULATION_STEPS = 1
        self.VALID_BATCH_SIZE = 32
        self.EPOCHS = epochs
        if self.EPOCHS != 3:
            self.MODEL_SAVE_DIR += f"_{self.EPOCHS}ep"
        self.MAX_GRAD_NORM = 1.0
        self.n_worker_train = 16
        self.cat_n_layers = cat_n_layers
        if self.cat_n_layers == 3:
            self.MODEL_SAVE_DIR += f"_{self.cat_n_layers}cat"
        self.froze_n_layers = froze_n_layers
        if froze_n_layers >= 0:
            self.MODEL_SAVE_DIR += f"_{froze_n_layers}froze"
        self.warmup_iters = warmup_samples//batch_size
        if self.warmup_iters > 0:
            self.MODEL_SAVE_DIR += f"_{warmup_samples}warm"
        self.frozen_warmup = frozen_warmup
        if frozen_warmup:
            # assert froze_n_layers >= 0
            self.MODEL_SAVE_DIR += f"_fwarm"

        self.warmup_scheduler = warmup_scheduler
        if warmup_scheduler != 'linear':
            self.MODEL_SAVE_DIR += '_cos'

        self.mask_pad_loss = mask_pad_loss

        self.fp16 = fp16
        if fp16:
            self.MODEL_SAVE_DIR += f"_fp16"

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
        elif self.model_type == 'albert':
            self.ALBERT_PATH = "/mfs/fangzhiqiang/nlp_model/albert-base-v2"
            self.TOKENIZER = SentencePieceTokenizer(self.ALBERT_PATH)
            self.model_config = transformers.AlbertConfig.from_pretrained(self.ALBERT_PATH)
            self.model_config.output_hidden_states = True

    def print_info(self):
        print(f"device:\t{os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Seed\t: {self.seed}")
        print(f"Learning Rate\t: {self.lr}")
        print(f"Batch Size:\t {self.TRAIN_BATCH_SIZE}")
        print(f"Max Length:\t{self.MAX_LEN}")
        print(f"Alpha\t: {self.alpha}")
        print(f"model: {self.model_type}")
        print(f"model_save_dir: {self.MODEL_SAVE_DIR}")
        print(f"train_dir: {self.TRAINING_FILE}")
