# coding=utf-8
"""
中文版本 layoutlm(v1) 从中文版本roberta进行初始化 然后做预训练
在原有的torch 版本 wwm roberta进行修改  当前版本实现支持 wwm
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import BertTokenizer
import logging
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
    DataCollatorForWholeWordMask
)
from libs.lm.models.CustomLayoutLMModel import CustomLayoutLMModel
from libs.lm.datasets.CnPretrainDataset import CnPretrainDataset
from libs.lm.collators.custom_data_collators import LayoutLmDataCollatorForLanguageModeling
from libs.lm.trainers.Trainers import CustomEvaluateTrainer

import kp_setup

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
df_train_data_file = os.path.join(kp_setup.data_dir, 'datasets', 'cn_pretrain', 'train.txt')
# todo: eval 怎么做的来着 ? 也是随机?
df_eval_data_file = os.path.join(kp_setup.data_dir, 'datasets', 'cn_pretrain', 'eval.txt')
# todo: eval 怎么做的来着 ? 也是随机?
df_base_model_path = '/home/ana/data2/models/chinese-roberta-wwm-ext'
df_do_train = True
df_do_eval = True
df_train_epochs = 3
# 自己手写一份layoutlm 的config
df_layoutlm_config_path = '/home/ana/data2/models/chinese-roberta-wwm-ext-layoutlm/config.json'

df_output_dir = os.path.join(kp_setup.output_dir, 'layout-wwm-debug')



@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default=df_output_dir,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(default=df_train_epochs,
                                    metadata={"help": "Total number of training epochs to perform."})
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )
    warmup_steps: int = field(default=10, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    do_train: bool = field(default=df_do_train, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=df_do_eval, metadata={"help": "Whether to run eval on the dev set."})
#     # 多搬运一些 不容易 GPU 爆内存
#     eval_accumulation_steps: Optional[int] = field(
#         default=1,
#         metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
#     )
    evaluation_strategy: str = field(
        default='epoch',
        metadata={"help": "The evaluation strategy to use."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=df_base_model_path,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    layoutlm_config_path: Optional[str] = field(
        default=df_layoutlm_config_path
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=df_train_data_file, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=df_eval_data_file,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    mlm: bool = field(
        default=True, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    block_size: int = field(
        default=512,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(model_args, data_args, training_args, tokenizer: PreTrainedTokenizer, evaluate=False):
    if evaluate == True:
        mode = 'eval'
        file_path = data_args.train_data_file
    else:
        mode = 'train'
        file_path = data_args.eval_data_file
    local_rank = training_args.local_rank
    max_seq_len = data_args.block_size
    overwrite_cache = data_args.overwrite_cache
    model_type = model_args.model_type
    # 注意 CnPretrainDatasete 在 model_type 为layoutlm 时，动作和bert一样，在中文情况符合预期，但是不合适用在英文
    return CnPretrainDataset(file_path, tokenizer, mode, model_type, local_rank, max_seq_len, overwrite_cache=True,
                             max_count=-1)


# 自定义eval
def custom_compute_metrics(eval_preds):
    # eval 环节容易爆内存  所以不启用这个
    # 要处理好 -100 的事情 仅仅关注 label ！= -100 的标签, 可以参考一下 loss的计算;
    # 但是如果全部结果都传到这边 可能会导致内存爆?
    preds, labels = eval_preds
    # 处理好 -100
    vocab_size = preds.shape[-1]
    exp_labels = np.expand_dims(labels, -1)
    exp_preds = preds[np.broadcast_to(exp_labels, preds.shape) != -100].reshape(-1, vocab_size)
    preds = np.argmax(exp_preds, axis=-1)
    labels = labels[labels != -100]
    accuracy = accuracy_score(labels, preds)
    # micro_f1 = round(f1_score(labels, preds, average="micro"), 3)
    # macro_f1 = round(f1_score(labels, preds, average="macro"), 3)
    # metrics = evaluate.load("accuracy")
    # return {"Accuracy": accuracy, "Micro F1": micro_f1, "Macro F1": macro_f1}
    return {"accuracy": accuracy}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.info(model_args)
    logging.info(data_args)
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logging.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # 起model:
    layoutlm_config_path = model_args.layoutlm_config_path
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = CustomLayoutLMModel.init_from_bert(model_args.model_name_or_path, layoutlm_config_path)

    if model.config.model_type in ["layoutlm", "bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        # data_args.block_size = min(data_args.block_size, tokenizer.max_len)
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets
    logging.info('data args--------------------\n{}'.format(data_args))
    train_dataset = get_dataset(model_args, data_args, training_args,
                                tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(model_args, data_args, training_args, tokenizer=tokenizer,
                               evaluate=True) if training_args.do_eval else None

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = CustomEvaluateTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # 默认实现容易爆内存
        # compute_metrics=custom_compute_metrics
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logging.info("*** Evaluate ***")
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
