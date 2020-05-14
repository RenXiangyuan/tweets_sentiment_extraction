# encoding: utf-8
"""
@author: renxiangyuan
@contact: xiangyuan_ren@shannonai.com
@file: utils.py
@time: 2020-04-16 11:54
"""
import random
import numpy as np
import torch


def calculate_jaccard_score(
        original_tweet,
        target_string,
        sentiment_val,
        idx_start,
        idx_end,
        offsets,
        verbose=False):
    if idx_end < idx_start:
        idx_end = idx_start
        # idx_start = idx_end
        # return jaccard(target_string.strip(), original_tweet.strip()), original_tweet
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    if sentiment_val != "neutral" and verbose == True:
        if filtered_output.strip().lower() != target_string.strip().lower():
            print("********************************")
            print(f"Output= {filtered_output.strip()}")
            print(f"Target= {target_string.strip()}")
            print(f"Tweet= {original_tweet.strip()}")
            print("********************************")

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def post_process(selected):
    return " ".join(set(selected.lower().split()))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.avg = 0.0

    def update(self, num, size=1):
        self.avg = (self.avg * self.count + num * size) / (self.count + size)
        self.count += size

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

hardcode_dict = {
    '6dbdb64223': 'wish',
    '96ff964db0': 'hate',
    '130945d03c': 'lol',
    '7375cfdd5b': 'blessings',
    '70a0bcd877': 'sad',
    'd6a572f589': 'fab',
    '4b3fd4ec22': 'sad',
    'df398a774e': 'fun',
    '3a906c871f': 'sad.',
    '12f21c8f19': 'LOL',
    'af3fed7fc3': 'miss',
    '1760ca7893': 'best',
    '2863f435bd': 'A little happy',
    'ce69e99e71': 'I`m not sleeping at all',
    'a54d3c2825': 'worth',
    '322b61740c': 'perfect',
    '697887b4a1': 'nice',
    'ee9df322d1': 'Sorry',
    '72cfb17265': 'fail',
    '03f9f6f798': 'I don`t think I`ve ever been so tierd in my life.',
    '8a8c28f5ba': 'amazing',
    '31fa81e0ae': 'fun',
    '19d585c61b': 'sorry',
    '568ad4a905': 'Happy',
    'c400a6bf99': 'wish I could go too',
    '3d1318d372': 'yes now would be good',
    'e9c337f756': 'thanks to you',
    '5419aaf31e': 'nice',
    'ad94c81511': 'hurt',
    'adac9ee2e1': 'so good',
    '915c66ead8': 'I don`t want her dressed up though',
    'ad7be4d16e': 'nice',
    '26dfa4924b': 'Happy',
    '37e710afb0': 'almost better than',
    'e668df7ceb': 'nice',
    'cd5989172a': 'Sorry',
    '2225e0fa43': 'so sad',
    '09d0f8f088': 'wow',
    'ee5eb5337b': 'sad',
    '654d710fce': 'hate',
    '7972092a15': 'Eww',
    '7c1d73feef': 'Blessings and Joy',
    'c1c67d1a99': 'SMILE~life is good!',
    '89545ebb49': 'Have a very Happy Mother`s Day!',
    # '8d91c2e24a': 'Happy',
    # '8e4bd833da': 'Happy',
    # 'da48252d73': 'happy',
    # '271f782910': 'Happy',
    # '29afbab19c': 'Happy',
    # '9d786a1526': 'Happy',
    '2780126312': 'Thanks!',
    '1211a1d91f': 'too bad',
    '18180bb2ec': 'Its so pretty',
    '9df7f02404': 'FunForBunny',
}