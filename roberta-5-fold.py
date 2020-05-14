# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tokenizers
import torch
from torch.utils.data import DataLoader
import transformers
import torch.nn as nn
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import jaccard, AverageMeter, EarlyStopping, calculate_jaccard_score
from data import TweetDataset
from lovasz import lovasz_hinge


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        if config.model_type == 'roberta':
            self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        elif config.model_type == 'electra':
            self.electra = transformers.ElectraModel.from_pretrained(config.ELECTRA_PATH, config=conf)
        else:
            raise NotImplementedError(f"{config.model_type} 不支持")

        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(conf.hidden_size * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        if config.do_IO:
            self.token_dropout = nn.Dropout(0.1)
            if config.loss_type == "lovasz":
                self.token_classifier = nn.Linear(conf.hidden_size, 1)
            elif config.loss_type == 'bce':
                self.token_classifier = nn.Linear(conf.hidden_size, 2)
            else:
                raise NotImplementedError(f"IO LOSS {config.loss_type} Invalid")

            torch.nn.init.normal_(self.token_classifier.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        if config.model_type == 'roberta':
            sequence_output, _, out = self.roberta(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        elif config.model_type == 'electra':
            sequence_output, _, out = self.electra(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        # out = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)[-1]

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not config.do_IO:
            return start_logits, end_logits

        sequence_output = self.token_dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)
        return start_logits, end_logits, token_logits


def loss_fn(start_logits, end_logits, start_positions, end_positions,
            token_logits=None, token_labels=None, mask=None):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    if not config.do_IO:
        return total_loss

    if config.loss_type == "lovasz":
        token_loss = token_lovasz_fn(token_logits, token_labels, mask)
    elif config.loss_type == 'bce':
        token_loss = token_loss_fn(token_logits, token_labels, mask)
    else:
        raise NotImplementedError(f"IO LOSS {config.loss_type} Invalid")

    return total_loss + config.alpha * token_loss


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


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

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
        if not config.do_IO:
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
        else:
            token_logits = model_out[2]
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end,
                           token_logits, labels, mask)

        if config.ACCUMULATION_STEPS > 1:
            loss = loss / config.ACCUMULATION_STEPS
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


def eval_fn(data_loader, model, device):
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
            if not config.do_IO:
                loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
            else:
                token_logits = model_out[2]
                loss = loss_fn(start_logits, end_logits, targets_start, targets_end,
                           token_logits, labels, mask)
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

    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        config=config,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

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

    device = torch.device("cuda")
    model = TweetModel(conf=config.model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.lr, eps=config.eps) # , eps=1e-8
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    es = EarlyStopping(patience=2, mode="max")
    print(f"{'-'*10}\nTraining is Starting for fold={fold}")

    # I'm training only for 3 epochs even though I specified 5!!!
    jaccard_list = []
    for epoch in range(config.EPOCHS):
        model_save_dir = os.path.join(config.MODEL_SAVE_DIR, f'model_{fold}_epoch_{epoch+1}.pth')
        if os.path.exists(model_save_dir):
            print(f"PTH已存在:{model_save_dir}")
            model.load_state_dict(torch.load(model_save_dir))
            continue
        print(f"\t\nEpoch:{epoch}")
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        jaccard_list.append(jaccard)
        print(f"Jaccard Score = {jaccard}")
        torch.save(model.state_dict(), model_save_dir)
        es(jaccard, model, model_path=os.path.join(config.MODEL_SAVE_DIR, f'model_{fold}.pth'))
        if es.early_stop:
            print("Early stopping")
            break

    print("Jarccard Scores:", jaccard_list)
    return jaccard_list


def ensemble_infer(model_paths=[]):
    if not model_paths:
        assert all(os.path.exists(p) for p in model_paths), f"model_paths不合法"

    df_test = pd.read_csv("/mfs/renxiangyuan/tweets/data/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values
    device = torch.device("cuda")

    model1 = TweetModel(conf=config.model_config)
    model1.to(device)
    if not model_paths:
        model1.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_0.pth")))
    else:
        model1.load_state_dict(torch.load(model_paths[0]))
    model1.eval()

    model2 = TweetModel(conf=config.model_config)
    model2.to(device)
    if not model_paths:
        model2.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_1.pth")))
    else:
        model2.load_state_dict(torch.load(model_paths[1]))
    model2.eval()

    model3 = TweetModel(conf=config.model_config)
    model3.to(device)
    if not model_paths:
        model3.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_2.pth")))
    else:
        model3.load_state_dict(torch.load(model_paths[2]))
    model3.eval()

    model4 = TweetModel(conf=config.model_config)
    model4.to(device)
    if not model_paths:
        model4.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_3.pth")))
    else:
        model4.load_state_dict(torch.load(model_paths[3]))
    model4.eval()

    model5 = TweetModel(conf=config.model_config)
    model5.to(device)
    if not model_paths:
        model5.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, "model_4.pth")))
    else:
        model5.load_state_dict(torch.load(model_paths[4]))
    model5.eval()

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

            outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
            outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
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
                    offsets=offsets[px]
                )
                final_output.append(output_sentence)

    sample = pd.read_csv("/mfs/renxiangyuan/tweets/data/sample_submission.csv")
    sample.loc[:, 'selected_text'] = final_output
    # sample.selected_text = sample.selected_text
    sub_save_dir=config.MODEL_SAVE_DIR +"/submission.csv"
    sample.to_csv(sub_save_dir, index=False)
    print("Submission Saved:", sub_save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--cuda_device",
        default='9',
        type=str,
        required=False,
        help="Which GPU To Use",
    )

    parser.add_argument(
        "--lr",
        default=5,
        type=int,
        required=False,
        help="Learning Rate(1e-5)",
    )
    args = parser.parse_args()
    args.cuda_device='9'
    args.lr = 10

    import os
    from utils import set_seed

    class Config(object):
        def __init__(self, train_dir, model_save_dir, batch_size=128, seed=42, lr=3e-5, model_type='roberta', alphe=0.3, do_IO=False, smooth=0):
            self.seed=seed
            self.lr = lr
            self.model_type = model_type
            self.TRAINING_FILE = train_dir  # '/data/nfs/fangzhiqiang/nlp_data/tweet_extraction/folds/train_folds.csv'# ak数据
            self.alpha = alphe
            self.do_IO = do_IO
            self.TRAIN_BATCH_SIZE = batch_size  # 16
            self.MODEL_SAVE_DIR = model_save_dir + f'/{round(self.lr*1e5)}e-05lr_{batch_size}bs_{self.seed}sd'
            self.smooth = smooth
            if smooth > 0:
                self.MODEL_SAVE_DIR += f"_{smooth}ls"
            if self.do_IO:
                self.MODEL_SAVE_DIR += f"_{alphe}alpha"


            self.MAX_LEN = 128
            # self.loss_type = 'bce'
            self.loss_type = 'lovasz'
            self.eps=1e-6
            self.ACCUMULATION_STEPS = 1
            self.VALID_BATCH_SIZE = 16
            self.EPOCHS = 3
            self.MAX_GRAD_NORM = 1.0

            if self.model_type == 'roberta':
                self.ROBERTA_PATH = "/mfs/fangzhiqiang/nlp_model/roberta-base"
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
            print(f"LR\t: {self.lr}")
            print(f"Alpha\t: {self.alpha}")
            print(f"model: {self.model_type}")
            print(f"model_save_dir: {self.MODEL_SAVE_DIR}")
            print(f"train_dir: {self.TRAINING_FILE}")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    config = Config(
        # train_dir= '/home/renxiangyuan/NLP/data/clean_data/train_filter_5.csv',
        train_dir='/mfs/renxiangyuan/tweets/data/train_folds.csv',
        model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-5-fold-ak',  # 基于ak数据训
        # model_save_dir = '/mfs/renxiangyuan/tweets/output/roberta-base-multi-lovasz-smooth-5-fold-ak',  # 基于ak数据训
        seed=42,
        lr=args.lr * 1e-5, # 9e-5,
        # model_type='electra'
        model_type='roberta',
        alphe=0.5,
        do_IO=True,
    )
    config.print()
    set_seed(config.seed)
    mode = "train"

    # 训练
    if mode == "train":
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        jaccard_scores = []
        for i in range(5):
            jaccard_scores.append(train(fold=i, config=config))
        for i, res_i in enumerate(jaccard_scores):
            print(i, res_i)
        print("mean", np.mean([max(scores) for scores in jaccard_scores]))
        config.print()

    # 测试
    # model_paths = [
    #     "/data/nfs/renxiangyuan/tweets/result-modmodel/roberta-base-5-fold-ak/2e-05lr_69sd/model_0_epoch_3.pth",
    #     "/data/nfs/renxiangyuan/tweets/result-modmodel/roberta-base-5-fold-ak/3e-05lr_42sd/model_1_epoch_2.pth",
    #     "/data/nfs/renxiangyuan/tweets/result-modmodel/roberta-base-5-fold-ak/3e-05lr_5845sd/model_2_epoch_3.pth",
    #     "/data/nfs/renxiangyuan/tweets/result-modmodel/roberta-base-5-fold-ak/2e-05lr_69sd/model_3_epoch_2.pth",
    #     "/data/nfs/renxiangyuan/tweets/result-modmodel/roberta-base-5-fold-ak/3e-05lr_5845sd/model_4_epoch_2.pth",
    # ]
    # ensemble_infer(model_paths)
    # ensemble_infer()

    # # 评估
    if mode=="evaluate":
        device = torch.device("cuda")
        model = TweetModel(conf=config.model_config)
        model.to(device)
        res = np.zeros((5,2))
        for fold in range(5):
            dfx = pd.read_csv(config.TRAINING_FILE)
            df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

            valid_dataset = TweetDataset(
                tweet=df_valid.text.values,
                sentiment=df_valid.sentiment.values,
                selected_text=df_valid.selected_text.values,
                config=config,
            )

            valid_data_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.VALID_BATCH_SIZE,
                num_workers=2
            )

            state_dict_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{fold}_epoch_2.pth")
            print(state_dict_dir)
            model.load_state_dict(torch.load(state_dict_dir))
            model.eval()

            jaccards = eval_fn(valid_data_loader, model, device)
            print(jaccards)
            res[fold][0] = jaccards

            state_dict_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{fold}_epoch_3.pth")
            print(state_dict_dir)
            model.load_state_dict(torch.load(state_dict_dir))
            model.eval()

            jaccards = eval_fn(valid_data_loader, model, device)
            print(jaccards)
            res[fold][1] = jaccards
        for i, res_i in enumerate(res):
            print(i, res_i[0], res_i[1])
        print("mean", np.mean([max(scores) for scores in res]))