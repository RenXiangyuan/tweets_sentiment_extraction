# encoding: utf-8
"""
@author: renxiangyuan
@contact: xiangyuan_ren@shannonai.com
@file: data.py
@time: 2020-04-16 11:54
"""

import torch
import numpy as np


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, config, multi_sentiment_cls=None):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        if 'roberta' in config.model_type:
            self.process_data = self.process_data_roberta
        elif config.model_type == 'electra':
            self.process_data = self.process_data_electra
        elif config.model_type == 'bart':
            self.process_data = self.process_data_bart
        elif 'albert' in config.model_type:
            self.process_data = self.process_data_albert
        else:
            raise NotImplementedError(f"{config.model_type} 不支持")

        self.multi_sent_loss_ratio = config.multi_sent_loss_ratio
        self.d_multi_sent = config.multi_sent_class
        self.multi_sent_cls = multi_sentiment_cls

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len,
        )

        res = {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_orig': data['orig_orig'],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }

        if self.multi_sent_loss_ratio > 0:
            cls_label = self.d_multi_sent[self.multi_sent_cls[item]]
            res['cls_labels'] =  torch.tensor(cls_label, dtype=torch.long)

        return res

    @staticmethod
    def process_data_roberta(tweet, selected_text, sentiment, tokenizer, max_len):
        orig_tweet = str(tweet)
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        assert len(input_ids) == len(token_type_ids)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4

        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        labels = [0] * len(input_ids)
        for idx in range(targets_start, targets_end + 1):
            labels[idx] = 1

        return {
            'ids': input_ids,
            'mask': mask,
            "labels": labels,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'orig_orig': orig_tweet,
            'sentiment': sentiment,
            'offsets': tweet_offsets
        }

    @staticmethod
    def process_data_bart(tweet, selected_text, sentiment, tokenizer, max_len):
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        input_ids = [0] + [sentiment_id[sentiment]] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        assert len(input_ids) == len(token_type_ids)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
        targets_start += 3
        targets_end += 3

        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        labels = [0] * len(input_ids)
        for idx in range(targets_start, targets_end + 1):
            labels[idx] = 1

        return {
            'ids': input_ids,
            'mask': mask,
            "labels": labels,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets
        }

    @staticmethod
    def process_data_electra(tweet, selected_text, sentiment, tokenizer, max_len):
        len_st = len(selected_text)
        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids[1:-1]
        tweet_offsets = tok_tweet.offsets[1:-1]

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 3893,
            'negative': 4997,
            'neutral': 8699
        }

        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
        targets_start += 3
        targets_end += 3

        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets
        }

    @staticmethod
    def process_data_albert(tweet, selected_text, sentiment, tokenizer, max_len):
        len_st = len(selected_text)
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        input_ids_orig, tweet_offsets = tokenizer.encode(tweet)

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 2221,
            'negative': 3682,
            'neutral': 8387
        }

        input_ids = [2] + [sentiment_id[sentiment]] + [3] + input_ids_orig + [3]
        token_type_ids = [0] * 3 + [1] * (len(input_ids_orig) + 1)
        assert len(input_ids) == len(token_type_ids)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
        targets_start += 3
        targets_end += 3

        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        labels = [0] * len(input_ids)
        for idx in range(targets_start, targets_end + 1):
            labels[idx] = 1

        return {
            'ids': input_ids,
            'mask': mask,
            'labels': labels,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets
        }
