"""
中文预训练模型数据集
经过预处理后，生成一个大文件，然后用这个数据集加载
    支持whole word masking 的dataset
        1. jieba
        2. baidu wordtag (todo)

更好的实现参考： https://github.com/huggingface/transformers/blob/main/examples/legacy/run_chinese_ref.py
"""

import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import jieba


logger = logging.getLogger(__name__)


class CnWwmPretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, mode, model_type, local_rank, max_seq_len, overwrite_cache=True,
                 max_count=-1):
        if local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        data_dir = os.path.dirname(file_path)

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}".format(
                mode,
                str(max_seq_len),
            ),
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)
            examples = read_examples_from_file(data_dir, mode)
            if max_count > 0:
                examples = examples[:max_count]
            features = self.convert_examples_to_features(
                self,
                examples,
                max_seq_len,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
            )
            # if training_args.local_rank in [-1, 0]:
            # logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)

        if local_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        logging.info("DATASET examples COUNT: {}".format(len(examples)))

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def get_segments(self, input_str, engine='jieba'):
        """
            这个是按照DataCollator的要求来进行分词，按照分词结果来填充chinese_ref
        :param input_str:
        :param engine:
        :return:
        """
        chinese_ref = []
        index = 1
        if engine == 'jieba':
            seq_cws = jieba.cut(input_str)
        for seq in seq_cws:
            for i, word in enumerate(seq):
                if i > 0:
                    chinese_ref.append(index)
                index += 1
        return chinese_ref

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_bboxes[index],
        )
        """
        return {
            "input_ids": self.all_input_ids[index],
            "attention_mask": self.all_input_mask[index],
            "token_type_ids": self.all_segment_ids[index],
            "bbox": self.all_bboxes[index]
        }

    def convert_examples_to_features(
            self,
            examples,
            max_seq_length,
            tokenizer,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=0,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            cls_token_box=[0, 0, 0, 0],
            sep_token_box=[1000, 1000, 1000, 1000],
            pad_token_box=[0, 0, 0, 0],
            pad_token_segment_id=0,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens = []
            token_boxes = []
            for word, box in zip(example.words, example.boxes):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            # example 过长 那就直接截断
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0

            tokens += [sep_token]
            token_boxes += [sep_token_box]
            if sep_token_extra:
                tokens += [sep_token]
                token_boxes += [sep_token_box]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                token_boxes += [cls_token_box]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                token_boxes = [cls_token_box] + token_boxes
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                                     [0 if mask_padding_with_zero else 1] * padding_length
                             ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                token_boxes = ([pad_token_box] * padding_length) + token_boxes
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                token_boxes += [pad_token_box] * padding_length

            input_str = "".join(example.words)
            chinese_ref = self.get_segments(input_str)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_boxes) == max_seq_length


            if ex_index < 3:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", "".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    boxes=token_boxes,
                    chinese_ref=chinese_ref
                )
            )
        return features


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, boxes):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.boxes = boxes


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            boxes
    ):
        assert (
                0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.boxes = boxes


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        boxes = []
        for line in f:
            # 发现空行
            if line.strip() == "" or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            boxes=boxes
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
            else:
                splits = line.split("\t")
                assert len(splits) == 5
                words.append(splits[4])
                box = [int(pos) for pos in splits[0:4]]
                boxes.append(box)

        if words:
            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index),
                    words=words,
                    boxes=boxes
                )
            )
    return examples


