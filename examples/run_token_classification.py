"""
一个序列标注的分类模型样例
参考官方样例实现
        zhangzhi600@20221014

https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb
https://github.com/NielsRogge/unilm/tree/master/layoutlm/examples/seq_labeling
"""
from __future__ import absolute_import, division, print_function
import os
import sys

# 会因为放在example 路径了， 所以需要向上引用1级目录 放入path内才能import kp_setup
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import kp_setup

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# 直接import 官方版本的实现
# 当前中文版本不一样的核心仅仅在tokenizer， 由于要配合寿险版本的java tokenizer， 当前只能以字为单位
from transformers import LayoutLMConfig, LayoutLMModel, BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
    LayoutLMForMaskedLM, LayoutLMForTokenClassification
from libs.lm.datasets.CnTableTokenClsDataset import CnTableTokenClsDataset
from libs.utils import make_sure_dir_there

# 默认配置 方便调试
df_model_path = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/output/layout-wwm-debug/checkpoint-1002'
#  df_base_model_path = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/output/layout-wwm-debug/'
df_train_file = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/data/datasets/pseudo_table_token_cls/train.txt'
df_eval_file = '/home/ana/data1/jpt_ntbk_wk/zz/layoutLM/data/datasets/pseudo_table_token_cls/eval.txt'
df_output_dir = os.path.join(kp_setup.output_dir, "layoutlm_token_cls_demo")
df_num_train_epochs = 2
df_labels = ["O", "A", "B", "C"]
df_label_map = {i: label for i, label in enumerate(df_labels)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(  # noqa C901
        args, train_dataset, eval_dataset, model, tokenizer, labels, pad_token_label_id
):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir="runs/" + os.path.basename(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=None,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
                args.max_steps
                // (len(train_dataloader) // args.gradient_accumulation_steps)
                + 1
        )
    else:
        t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # inputs = {
            #     "input_ids": batch[0].to(args.device),
            #     "attention_mask": batch[1].to(args.device),
            #     "labels": batch[3].to(args.device),
            # }
            # if args.model_type in ["layoutlm"]:
            #     inputs["bbox"] = batch[4].to(args.device)
            # inputs["token_type_ids"] = (
            #     batch[2].to(args.device) if args.model_type in ["bert", "layoutlm"] else None
            # )  # RoBERTa don"t use segment_ids

            for key in batch:
                batch[key] = batch[key].to(args.device)
            inputs = batch

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                        args.local_rank in [-1, 0]
                        and args.logging_steps > 0
                        and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                            args.local_rank in [-1, 0] and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args,
                            eval_dataset,
                            model,
                            tokenizer,
                            labels,
                            pad_token_label_id,
                            mode="dev"
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                        args.local_rank in [-1, 0]
                        and args.save_steps > 0
                        and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # 每个epoch 也存一下
        if True:
            output_dir = os.path.join(
                args.output_dir, "checkpoint-epoch-{}".format(epoch)
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s", output_dir)

            # 每个epoch也测试一下
            if (
                    args.local_rank in [-1, 0] and args.evaluate_during_training
            ):  # Only evaluate when single GPU otherwise metrics may not average well
                results, _ = evaluate(
                    args,
                    eval_dataset,
                    model,
                    tokenizer,
                    labels,
                    pad_token_label_id,
                    mode="dev",
                    prefix="epoch-{}".format(epoch)
                )
                for key, value in results.items():
                    tb_writer.add_scalar(
                        "eval_epoch_{}".format(key), value, global_step
                    )
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar(
                "loss",
                (tr_loss - logging_loss) / args.logging_steps,
                global_step,
                )
            logging_loss = tr_loss

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )

    # Eval!
    logging.info("***** Running evaluation %s *****", prefix)
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # inputs = {
            #     "input_ids": batch[0].to(args.device),
            #     "attention_mask": batch[1].to(args.device),
            #     "labels": batch[3].to(args.device),
            # }
            # if args.model_type in ["layoutlm"]:
            #     inputs["bbox"] = batch[4].to(args.device)
            # inputs["token_type_ids"] = (
            #     batch[2].to(args.device)
            #     if args.model_type in ["bert", "layoutlm"]
            #     else None
            # )  # RoBERTa don"t use segment_ids
            for key in batch:
                batch[key] = batch[key].to(args.device)
            inputs = batch

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

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

    report = classification_report(out_label_list, preds_list)
    logging.info("\n" + report)

    logging.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def main():
    # ######## CONFIG #################
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="训练文件所在路径",
    )
    parser.add_argument(
        "--model_type",
        default='layoutlm',
        type=str,
        help="仅仅支持layoutlm",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=df_model_path,
        type=str,
        help="预训练完毕后模型的路径"
    )
    parser.add_argument(
        "--output_dir",
        default=df_output_dir,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", default=True, help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true",  default=True, help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        default=True,
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        default=True,
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=df_num_train_epochs,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=True,
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        default=True,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
    ):
        if not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )
        else:
            if args.local_rank in [-1, 0]:
                shutil.rmtree(args.output_dir)

    if not os.path.exists(args.output_dir) and (args.do_eval or args.do_predict):
        make_sure_dir_there(args.output_dir)
        logging.warning("mkdir: {}".format(args.output_dir))
    if (
            not os.path.exists(args.output_dir)
            and args.do_train
            and args.local_rank in [-1, 0]
    ):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # #################################
    if args.labels == '':
        # use default
        labels = df_labels
    else:
        labels = get_labels(args.labels)
    num_labels = len(labels)
    # -100 for default
    pad_token_label_id = CrossEntropyLoss().ignore_index

    if args.local_rank not in [0, -1]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizer
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logging.info("Training/evaluation parameters %s", args)

    eval_dataset = CnTableTokenClsDataset(df_eval_file, tokenizer, labels, pad_token_label_id, 'eval',
                                          args.model_type, args.local_rank, args.max_seq_length)
    # train here
    if args.do_train:
        train_dataset = CnTableTokenClsDataset(df_train_file, tokenizer, labels, pad_token_label_id, 'train',
                                               args.model_type, args.local_rank, args.max_seq_length)

        global_step, tr_loss = train(
            args, train_dataset, eval_dataset, model, tokenizer, labels, pad_token_label_id
        )
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logging.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logging.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(
                args,
                eval_dataset,
                model,
                tokenizer,
                labels,
                pad_token_label_id,
                mode="test",
                prefix=global_step,
            )
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    return results


if __name__ == '__main__':
    main()
