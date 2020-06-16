# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import transformers
import torch.nn as nn
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import AverageMeter, EarlyStopping, calculate_jaccard_score
from data import TweetDataset
from lovasz import lovasz_hinge

from utils import clean_item
from utils import filter_set

from apex import amp

class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf, config):
        super(TweetModel, self).__init__(conf)
        self.config = config
        if 'roberta' in config.model_type:
            self.model = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        # elif config.model_type == 'electra':
        #     self.electra = transformers.ElectraModel.from_pretrained(config.ELECTRA_PATH, config=conf)
        # elif config.model_type == 'bart':
        #     self.bart = transformers.BartModel.from_pretrained(config.BART_PATH, config=conf)
        elif 'albert' in config.model_type:
            self.model = transformers.AlbertModel.from_pretrained(config.ALBERT_PATH, config=conf)
        else:
            raise NotImplementedError(f"{config.model_type} 不支持")

        if config.frozen_warmup and config.warmup_iters > 0:
            self.frozen(12)
        elif config.froze_n_layers >= 0:
            self.frozen(config.froze_n_layers)

        self.drop_out = nn.Dropout(0.1)

        if config.conv_head:
            self.head = nn.Conv1d(conf.hidden_size * config.cat_n_layers, 2, kernel_size=5,
                                  stride=1, padding=2, bias=False)
            torch.nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="sigmoid")
        else:
            self.head = nn.Linear(conf.hidden_size * config.cat_n_layers, 2)
            torch.nn.init.normal_(self.head.weight, std=0.02)

        if config.multi_sent_loss_ratio > 0:
            self.sent_dropout = nn.Dropout(0.1)
            self.sent_classifier = nn.Linear(conf.hidden_size, len(config.multi_sent_class))
            torch.nn.init.normal_(self.sent_classifier.weight, std=0.02)

        if config.io_loss_ratio > 0:
            self.token_dropout = nn.Dropout(0.1)
            if config.io_loss_type == "lovasz":
                self.token_classifier = nn.Linear(conf.hidden_size, 1)
            elif config.io_loss_type == 'bce':
                self.token_classifier = nn.Linear(conf.hidden_size, 2)
            else:
                raise NotImplementedError(f"IO LOSS {config.io_loss_type} Invalid")

            torch.nn.init.normal_(self.token_classifier.weight, std=0.02)

    def save_head(self, save_path):
        state_dict = self.state_dict()
        head_state_dict = {k: v for k, v in state_dict.items() if k.startswith("head.")}
        torch.save(head_state_dict, save_path)

    def load_head(self, head_path):
        state_dict = self.state_dict()
        head_state_dict = torch.load(head_path)
        state_dict.update(head_state_dict)
        self.load_state_dict(state_dict)

    def frozen(self, froze_n_layers):
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        print(f"{self.config.model_type} Embedding Frozen")
        for i in range(froze_n_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False
            print(f"{self.config.model_type} Encoder Layer {i} Frozen")

    def unfrozen(self, frozen_n_layers):
        for param in self.model.encoder.layer[11].parameters():
            if param.requires_grad:
                return
            break
        for i in range(frozen_n_layers, 12):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
            print(f"{self.config.model_type} Encoder Layer {i} Unfrozen")

    def forward(self, ids, mask, token_type_ids):
        if 'roberta' in self.config.model_type:
            sequence_output, _, out = self.model(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        elif self.config.model_type == 'electra':
            sequence_output, _, out = self.electra(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        elif self.config.model_type == 'bart':
            sequence_output, out, _, _ = self.bart(
                ids,
                attention_mask=mask,
                decoder_attention_mask=mask,
                # token_type_ids=token_type_ids
            )
        elif 'albert' in self.config.model_type:
            sequence_output, _, out = self.model(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        # out = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)[-1]

        if self.config.cat_n_layers == 1:
            out = out[-1]
        elif self.config.cat_n_layers == 2:
            out = torch.cat((out[-1], out[-2]), dim=-1)
        elif self.config.cat_n_layers == 3:
            out = torch.cat((out[-1], out[-2], out[-3]), dim=-1)
        else:
            raise NotImplementedError()

        out = self.drop_out(out)
        if self.config.conv_head:
            out = out.transpose(2, 1)
            logits = self.head(out)
            logits = logits.transpose(2, 1)
        else:
            logits = self.head(out)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.config.multi_sent_loss_ratio > 0:
            cls_hidden_state = sequence_output[:, 0, :]
            cls_hidden_state = self.sent_dropout(cls_hidden_state)
            cls_logit = self.sent_classifier(cls_hidden_state)
            return start_logits, end_logits, cls_logit

        if self.config.io_loss_ratio > 0:
            sequence_output = self.token_dropout(sequence_output)
            token_logits = self.token_classifier(sequence_output)
            return start_logits, end_logits, token_logits

        return start_logits, end_logits

def loss_fn(start_logits, end_logits, start_positions, end_positions,
            cls_logit=None, cls_label=None, token_logits=None, token_labels=None, mask=None, config=None):

    if config.mask_pad_loss:
        start_logits[:, :4] -= 100
        end_logits[:, :4] -= 100
        start_logits -= (mask == 0) * 1e4
        end_logits -= (mask == 0) * 1e4

    if config.smooth > 0:
        one_hot_start_label = torch.zeros_like(start_logits).scatter(1, start_positions.view(-1, 1), 1)
        one_hot_end_label = torch.zeros_like(start_logits).scatter(1, end_positions.view(-1, 1), 1)
        one_hot_start_label = one_hot_start_label * (1 - config.smooth) + \
                              (1 - one_hot_start_label) * config.smooth / (config.MAX_LEN - 1)
        one_hot_end_label = one_hot_end_label * (1 - config.smooth) + \
                              (1 - one_hot_end_label) * config.smooth / (config.MAX_LEN - 1)
        if config.mask_pad_loss:
            one_hot_start_label[mask == 0] = 0
            one_hot_end_label[mask == 0] = 0
            one_hot_start_label[:, :4] = 0
            one_hot_end_label[:, :4] = 0

        start_loss = -torch.sum(F.log_softmax(start_logits, dim=1) * one_hot_start_label)/start_logits.size(0)
        end_loss = -torch.sum(F.log_softmax(end_logits, dim=1) * one_hot_end_label)/start_logits.size(0)
    else:
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    if config.multi_sent_loss_ratio > 0:
        cls_loss_fct = nn.CrossEntropyLoss()
        multi_sent_loss = cls_loss_fct(cls_logit, cls_label)
        # print(total_loss, multi_sent_loss)
        return total_loss + config.multi_sent_loss_ratio * multi_sent_loss

    if not config.io_loss_ratio > 0:
        return total_loss

    if config.io_loss_type == "lovasz":
        token_loss = token_lovasz_fn(token_logits, token_labels, mask)
    elif config.io_loss_type == 'bce':
        token_loss = token_loss_fn(token_logits, token_labels, mask)
    else:
        raise NotImplementedError(f"IO LOSS {config.io_loss_type} Invalid")

    return total_loss + config.io_loss_ratio * token_loss


def token_lovasz_fn(token_logits, token_labels, mask):
    loss = lovasz_hinge(token_logits * mask.unsqueeze(2), token_labels * mask)
    return loss


def token_loss_fn(token_logits, token_labels, mask):
    loss_fct = nn.CrossEntropyLoss()
    # Only keep active parts of the loss
    if mask is not None:
        active_loss = mask.view(-1) == 1
        active_logits = token_logits.view(-1, 2)
        active_labels = torch.where(
            active_loss, token_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(token_labels)
        )
        loss = loss_fct(active_logits, active_labels)
    else:
        loss = loss_fct(token_logits.view(-1, 2), token_labels.view(-1))

    return loss


def train_fn(data_loader, model, optimizer, device, config, scheduler=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        # TODO: 没有考虑 不froze 情况的unfrozen
        if 0 < config.warmup_iters == bi \
                and config.frozen_warmup and config.froze_n_layers >= 0:
            model.unfrozen(config.froze_n_layers)

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        labels = d["labels"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)

        model.zero_grad()

        model_out = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        start_logits, end_logits = model_out[0], model_out[1]

        if config.multi_sent_loss_ratio > 0:
            cls_labels = d["cls_labels"].to(device, dtype=torch.long)
            cls_logits = model_out[2]
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end, cls_logits, cls_labels, config=config)
        elif not config.io_loss_ratio > 0:
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end, mask=mask, config=config)
        else:
            token_logits = model_out[2]
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end,
                           token_logits=token_logits, token_labels=labels, mask=mask, config=config)

        if config.ACCUMULATION_STEPS > 1:
            loss = loss / config.ACCUMULATION_STEPS
        if config.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if config.ACCUMULATION_STEPS == 1 or config.ACCUMULATION_STEPS > 1 and (bi + 1) % config.ACCUMULATION_STEPS == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

        outputs_start = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def eval_fn(data_loader, model, device, config):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            labels = d["labels"]
            offsets = d["offsets"].numpy()
            orig_orig = d['orig_orig']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            model_out= model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            start_logits, end_logits = model_out[0], model_out[1]
            if len(model_out) > 2:
                io_logits = model_out[2]
                io_probs = F.sigmoid(io_logits).detach().cpu().numpy()
            else:
                io_probs = None




            if config.multi_sent_loss_ratio > 0:
                cls_logits = model_out[2]
                cls_labels = d["cls_labels"].to(device, dtype=torch.long)
                loss = loss_fn(start_logits, end_logits, targets_start, targets_end, cls_logits, cls_labels, config=config)
            elif not config.io_loss_ratio > 0:
                loss = loss_fn(start_logits, end_logits, targets_start, targets_end, config=config)
            else:
                token_logits = model_out[2]
                loss = loss_fn(start_logits, end_logits, targets_start, targets_end,
                           token_logits=token_logits, token_labels=labels, mask=mask, config=config)
            outputs_start = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                if io_probs is not None:
                    io_prob = io_probs[px].squeeze()
                else:
                    io_prob = None
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px],
                    io_prob=io_prob,
                    orig_orig=orig_orig[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    return jaccards.avg


def train(fold, config):
    dfx = pd.read_csv(config.TRAINING_FILE)

    # from utils import hardcode_dict
    # for i in range(len(dfx)):
    #     id_ = dfx['textID'][i]
    #     if id_ in hardcode_dict:
    #         dfx.loc[i, 'selected_text'] = hardcode_dict[id_]

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    np_train = np.array(df_train)

    if config.clean_data:
        # 若要不训练neutral
        # np_train_idx = [i for i, item_i in enumerate(np_train) if item_i[3] != 'neutral']
        # np_train = np_train[np_train_idx]

        # 过滤outlier
        np_train_idx = [i for i, item_i in enumerate(np_train) if item_i[0] not in filter_set]
        np_train = np_train[np_train_idx]

        # 修正input label
        for i, item in enumerate(np_train):
            # if item[3] == 'neutral':
            #     continue
            np_train[i] = clean_item(item)

    if config.shuffle_seed == -1:
        train_dataset = TweetDataset(
            # tweet=df_train.text.values,
            # sentiment=df_train.sentiment.values,
            # selected_text=df_train.selected_text.values,
            tweet=np_train[:, 1],
            sentiment=np_train[:,3],
            selected_text=np_train[:,2],
            config=config,
            multi_sentiment_cls = df_train.extra_sentiment.values if config.multi_sent_loss_ratio > 0 else None,
        )
    else:
        import sklearn
        index = np.arange(len(np_train))
        index = sklearn.utils.shuffle(index, random_state=config.shuffle_seed)
        assert len(set(index)) == len(np_train)
        train_dataset = TweetDataset(
            tweet=np_train[index][:, 1],
            sentiment=np_train[index][:, 3],
            selected_text=np_train[index][:, 2],
            config=config,
            multi_sentiment_cls=df_train.extra_sentiment.values if config.multi_sent_loss_ratio > 0 else None,
        )
    # train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        # sampler=train_sampler,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.n_worker_train,
    )

    np_valid = np.array(df_valid)
    if config.clean_data:
        # 若要不训练neutral
        # np_train_idx = [i for i, item_i in enumerate(np_train) if item_i[3] != 'neutral']
        # np_train = np_train[np_train_idx]

        # 过滤outlier
        np_valid_idx = [i for i, item_i in enumerate(np_valid) if item_i[0] not in filter_set]
        np_valid = np_valid[np_valid_idx]

        # 修正input label
        for i, item in enumerate(np_valid):
            if item[3] == 'neutral':
                continue
            np_valid[i] = clean_item(item)

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        config=config,
        multi_sentiment_cls=df_train.extra_sentiment.values if config.multi_sent_loss_ratio > 0 else None,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
    )

    device = torch.device("cuda")
    model = TweetModel(conf=config.model_config, config=config)
    if config.init_seed >= 0:
        model.load_head(f"/mfs/renxiangyuan/tweets/data/heads/{config.init_seed}/head.pth")
    model.to(device)

    num_train_steps = len(train_data_loader) // config.ACCUMULATION_STEPS * config.EPOCHS
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.lr, eps=config.eps) # , eps=1e-8

    if config.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if config.warmup_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_iters,
            num_training_steps=num_train_steps,
        )
    elif config.warmup_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_iters,
            num_training_steps=num_train_steps,
        ); print("Using Cosine Scheduler")
    else:
        raise NotImplementedError()

    es = EarlyStopping(patience=3, mode="max")
    print(f"{'-'*10}\nTraining is Starting for fold={fold}")

    jaccard_list = []
    # model.save_head(config.MODEL_SAVE_DIR+f"/fold{fold}_head.pth")
    for epoch in range(config.EPOCHS):
        model_save_dir = os.path.join(config.MODEL_SAVE_DIR, f'model_{fold}_epoch_{epoch+1}.pth')
        if os.path.exists(model_save_dir):
            print(f"PTH已存在:{model_save_dir}")
            model.load_state_dict(torch.load(model_save_dir))
            continue
        print(f"\t\nEpoch:{epoch}")
        train_fn(train_data_loader, model, optimizer, device, config, scheduler=scheduler)  # load schedular 会有不match
        jaccard = eval_fn(valid_data_loader, model, device, config)
        jaccard_list.append(jaccard)
        print(f"Jaccard Score = {jaccard}")
        torch.save(model.state_dict(), model_save_dir)
        es(jaccard, model, model_path=os.path.join(config.MODEL_SAVE_DIR, f'model_{fold}.pth'))
        if es.early_stop:
            print("Early stopping")
            break
    print("Jarccard Scores:", jaccard_list)
    return jaccard_list


def ensemble_infer(model_paths, config):
    if model_paths:
        assert all(os.path.exists(p) for p in model_paths), f"model_paths不合法"

    df_test = pd.read_csv("/mfs/renxiangyuan/tweets/data/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values
    device = torch.device("cuda")

    def eval_fold(fold_i, model):
        dfx = pd.read_csv(config.TRAINING_FILE)
        df_valid = dfx[dfx.kfold == fold_i].reset_index(drop=True)
        valid_dataset = TweetDataset(
            tweet=df_valid.text.values,
            sentiment=df_valid.sentiment.values,
            selected_text=df_valid.selected_text.values,
            config=config,
        )
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=2,
        )
        jaccard = eval_fn(valid_data_loader, model, device, config)
        print(f"Fold{fold_i}", jaccard)

    model1 = TweetModel(conf=config.model_config, config=config)
    model1.to(device)
    if not model_paths:
        model1.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_0.pth")))
    else:
        model1.load_state_dict(torch.load(model_paths[0]))
    model1.eval()
    eval_fold(0, model1)


    model2 = TweetModel(conf=config.model_config, config=config)
    model2.to(device)
    if not model_paths:
        model2.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_1.pth")))
    else:
        model2.load_state_dict(torch.load(model_paths[1]))
    model2.eval()
    eval_fold(1, model2)

    model3 = TweetModel(conf=config.model_config, config=config)
    model3.to(device)
    if not model_paths:
        model3.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_2.pth")))
    else:
        model3.load_state_dict(torch.load(model_paths[2]))
    model3.eval()
    eval_fold(2, model3)

    model4 = TweetModel(conf=config.model_config, config=config)
    model4.to(device)
    if not model_paths:
        model4.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_3.pth")))
    else:
        model4.load_state_dict(torch.load(model_paths[3]))
    model4.eval()
    eval_fold(3, model4)

    model5 = TweetModel(conf=config.model_config, config=config)
    model5.to(device)
    if not model_paths:
        model5.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_4.pth")))
    else:
        model5.load_state_dict(torch.load(model_paths[4]))
    model5.eval()
    eval_fold(4, model5)

    final_output = []

    test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        config=config,
    )

    data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            # targets_start = d["targets_start"]
            # targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()
            orig_orig = d['orig_orig']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            # targets_start = targets_start.to(device, dtype=torch.long)
            # targets_end = targets_end.to(device, dtype=torch.long)

            out1 = model1(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start1, outputs_end1 = out1[0], out1[1]

            out2 = model2(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start2, outputs_end2 = out2[0], out2[1]

            out3 = model3(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start3, outputs_end3 = out3[0], out3[1]

            out4 = model4(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start4, outputs_end4 = out4[0], out4[1]

            out5 = model5(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start5, outputs_end5 = out5[0], out5[1]

            # ensemble logits
            outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
            outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            # ensemble probs
            # outputs_starts = [torch.softmax(outputs_start, dim=1).cpu().detach().numpy() for outputs_start in [
            #     outputs_start1, outputs_start2, outputs_start3, outputs_start4, outputs_start5
            # ]]
            # outputs_ends = [torch.softmax(outputs_end, dim=1).cpu().detach().numpy() for outputs_end in [
            #     outputs_end1, outputs_end2, outputs_end3, outputs_end4, outputs_end5
            # ]]
            # outputs_start = np.mean(outputs_starts, axis=0)
            # outputs_end = np.mean(outputs_ends, axis=0)

            # jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px],
                    orig_orig=orig_orig
                )
                final_output.append(output_sentence)

    sample = pd.read_csv("/mfs/renxiangyuan/tweets/data/sample_submission.csv")
    sample.loc[:, 'selected_text'] = final_output
    # sample.selected_text = sample.selected_text
    sub_save_dir=config.MODEL_SAVE_DIR +"/submission.csv"
    sample.to_csv(sub_save_dir, index=False)
    print("Submission Saved:", sub_save_dir)

