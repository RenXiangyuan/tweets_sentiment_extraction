# encoding: utf-8
"""
@author: renxiangyuan
@contact: renxy.vertago@gmail.com
@file: model.py
@time: 2020-06-10 13:32
"""
import torch
import torch.nn as nn
import transformers


class SentimentPretrainModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(SentimentPretrainModel, self).__init__(config.model_config)
        self.config = config
        if 'roberta' in config.model_type:
            self.model = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=config.model_config)
        else:
            raise NotImplementedError(f"{config.model_type} 不支持")

        if config.frozen_warmup and config.warmup_iters > 0:
            self.frozen(12)
        elif config.froze_n_layers >= 0:
            self.frozen(config.froze_n_layers)

        self.drop_out = nn.Dropout(0.1)

        self.head = nn.Linear(config.model_config.hidden_size, 1, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)

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
            sequence_output, _, _ = self.model(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
        else:
            raise NotImplementedError(f"{self.config.model_type} 不支持")

        sentiment = sequence_output[:, 1:4, :]
        sentiment = self.drop_out(sentiment)
        sentiment = self.head(sentiment)
        sentiment = sentiment.squeeze(-1)

        return sentiment


def loss_fn(sentiment, label, config=None):
    loss_fct = nn.CrossEntropyLoss()
    sentiment_loss = loss_fct(sentiment, label)
    # sentiment_loss = loss_fct(sentiment.squeeze(-1), label.unsqueeze(1))

    return sentiment_loss
