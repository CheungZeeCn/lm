"""
一个序列标注的分类模型样例
"""

import sys
import os
from transformers import AdamW
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 会因为放在example 路径了， 所以需要向上引用1级目录 放入path内才能import kp_setup
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import kp_setup

import logging
from torch import nn
from transformers import LayoutLMConfig, LayoutLMModel, BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
    LayoutLMForMaskedLM, LayoutLMForTokenClassification
from libs.lm.datasets.CnTableTokenClsDataset import CnTableTokenClsDataset
from layoutlm.data.funsd import FunsdDataset, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

model_path = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/output/layout-wwm-debug/checkpoint-1002'
base_model_path = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/output/layout-wwm-debug/'
train_file = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/data/datasets/pseudo_table_token_cls/train.txt'
eval_file = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/data/datasets/pseudo_table_token_cls/eval.txt'
labels = ["O", "A", "B", "C"]
label_map = {i: label for i, label in enumerate(labels)}



def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def main():
    global labels
    pad_token_label_id = CrossEntropyLoss().ignore_index
    model = LayoutLMForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    train_dataset = CnTableTokenClsDataset(train_file, tokenizer, labels, pad_token_label_id, 'train', 'layoutlm', -1, 512)
    eval_dataset = CnTableTokenClsDataset(eval_file, tokenizer, labels, pad_token_label_id, 'eval', 'layoutlm', -1, 512)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=2)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=2)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    global_step = 0
    num_train_epochs = 2
    t_total = len(train_dataloader) * num_train_epochs  # total number of training steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # put the model in training mode
    model.train()

    for epoch in range(num_train_epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            #  print(batch.keys())
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss

            # print loss every 100 steps
            # if global_step % 100 == 0:
            #     print(f"Loss after {global_step} steps: {loss.item()}")

            # backward pass to get the gradients
            loss.backward()

            # print("Gradients on classification head:")
            # print(model.classifier.weight.grad[6,:].sum())

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)



if __name__ == '__main__':
    main()