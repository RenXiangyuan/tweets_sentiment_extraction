# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: train_eval.py
@time: 2020-06-10 13:54
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils import AverageMeter
from sentiment_pretrain.model import loss_fn, multi_sent_loss_fn
from data import TweetDataset
from sentiment_pretrain.model import SentimentPretrainModel
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import EarlyStopping
import os


def train_fn(data_loader, model, optimizer, device, config, scheduler=None):
    model.train()
    losses = AverageMeter()
    multi_sent_losses = AverageMeter()
    accuracies = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        # TODO: 没有考虑 不froze 情况的unfrozen
        if 0 < config.warmup_iters == bi \
                and config.frozen_warmup and config.froze_n_layers >= 0:
            model.unfrozen(config.froze_n_layers)

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment_orig = d["sentiment"]
        orig_tweet = d["orig_tweet"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        d_sentiment = {"positive": 0, "neutral": 1, "negative": 2}
        sentiment_np = [d_sentiment[sent] for sent in sentiment_orig]
        sentiment = torch.tensor(sentiment_np, dtype=torch.long).cuda()

        model.zero_grad()

        cls_logits, logits = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(logits, sentiment, config=config)

        if config.multi_sent_loss_ratio > 0:
            multi_sent = d["cls_labels"].to(device, dtype=torch.long)
            loss += multi_sent_loss_fn(cls_logits, multi_sent, config)


        if config.ACCUMULATION_STEPS > 1:
            loss = loss / config.ACCUMULATION_STEPS
        # if config.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        loss.backward()

        if config.ACCUMULATION_STEPS == 1 or config.ACCUMULATION_STEPS > 1 and (bi + 1) % config.ACCUMULATION_STEPS == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

        pred_sentiment = torch.max(logits, axis=1)[1].cpu().detach().numpy()
        acc = np.mean(sentiment_np == pred_sentiment)
        accuracies.update(acc, ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, acc=accuracies.avg)


def eval_fn(data_loader, model, device, config):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment_orig = d["sentiment"]
            orig_tweet = d["orig_tweet"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            d_sentiment = {"positive": 0, "neutral": 1, "negative": 2}
            sentiment_np = [d_sentiment[sent] for sent in sentiment_orig]
            sentiment = torch.tensor(sentiment_np, dtype=torch.long).cuda()

            logits= model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            loss = loss_fn(logits, sentiment, config=config)

            pred_sentiment = torch.max(logits, axis=1)[1].cpu().detach().numpy()
            acc = np.mean(sentiment_np == pred_sentiment)

            accuracies.update(acc, ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, accuracies=accuracies.avg)

    return accuracies.avg


def train(np_train, np_valid, config):
    fold = 0

    if config.shuffle_seed == -1:
        train_dataset = TweetDataset(
            # tweet=df_train.text.values,
            # sentiment=df_train.sentiment.values,
            # selected_text=df_train.selected_text.values,
            tweet=np_train[:, 1],
            selected_text=np_train[:,2],
            sentiment=np_train[:,3],
            config=config,
            multi_sentiment_cls = None if config.multi_sent_loss_ratio <= 0 else np_train[:,6],
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
            multi_sentiment_cls= None,
        )
    # train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        # sampler=train_sampler,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.n_worker_train,
    )


    valid_dataset = TweetDataset(
        tweet=np_valid[:, 1],
        sentiment=np_valid[:, 2],
        selected_text=np_valid[:, 1],
        config=config,
        multi_sentiment_cls= None,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
    )

    device = torch.device("cuda")
    model = SentimentPretrainModel(config=config)
    model.to(device)

    num_train_steps = len(train_data_loader) // config.ACCUMULATION_STEPS * config.EPOCHS
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.lr, eps=config.eps) # , eps=1e-8

    # if config.fp16:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

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

    print(f"{'-'*10}\nTraining Start")

    acc_list = []
    # model.save_head(config.MODEL_SAVE_DIR+f"/fold{fold}_head.pth")
    for epoch in range(config.EPOCHS):
        model_save_dir = os.path.join(config.MODEL_SAVE_DIR, f'model_{fold}_epoch_{epoch+1}.pth')
        if os.path.exists(model_save_dir):
            print(f"PTH已存在:{model_save_dir}")
            model.load_state_dict(torch.load(model_save_dir))
            continue
        print(f"\t\nEpoch:{epoch}")
        train_fn(train_data_loader, model, optimizer, device, config, scheduler=scheduler)  # load schedular 会有不match
        accuracies = eval_fn(valid_data_loader, model, device, config)
        acc_list.append(accuracies)
        print(f"Jaccard Score = {accuracies}")
        torch.save(model.state_dict(), model_save_dir)
    print("ACC Scores:", acc_list)
    return acc_list


