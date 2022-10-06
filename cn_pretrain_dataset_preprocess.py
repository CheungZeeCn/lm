"""
   假设输出是paddle ocr的格式， 将其转换为ChnPretrainDataset文件的格式;

"""

import os
import kp_setup
import logging
import tqdm
from libs.lm.datasets import utils as ds_utils
from libs import utils


def pa_preprocessed_format_to_layoutlm_format(lines, fn=""):
    ret = []
    for ori_line in lines:
        line = ori_line.strip()
        ls = ori_line.split("\t")
        if line != "":
            if len(ls) != 9:
                logging.warning("LINE FORMAT ERROR, IGNORE: [{}]:[{}]".format(fn, line))
                continue
            else:
                char_scaled_bbox = [int(ls[0]), int(ls[1]), int(ls[4]), int(ls[5])]
                ch = ls[9]
                chars_and_coords = [{"box": char_scaled_bbox, "text": ch}]
                scaled_chars_and_coords = ds_utils.rectify_chars_coords(chars_and_coords)
                for char_info in scaled_chars_and_coords:
                    bbox_string = "\t".join([str(x) for x in char_info['box']])
                    char_str = "{}\t{}".format(bbox_string, char_info['text'])
                    ret.append(char_str)
    return ret


def paddle_format_to_layoutlm_format(lines, fn=""):
    x_size, y_size = [float(digit_str) for digit_str in lines[0].strip().split("\t")]
    ret = []
    for line in lines[1:]:
        line = line.strip()
        ls = line.split("\t")
        if line != "":
            if len(ls) != 10:
                logging.warning("LINE FORMAT ERROR, IGNORE: [{}]:[{}]".format(fn, line))
                continue
            else:
                segment_actual_bbox = [float(ls[0]), float(ls[1]), float(ls[4]), float(ls[5])]
                text = ls[9]
                # 获取每个字的坐标
                chars_and_coords = ds_utils.cn_segment_coord_to_chars_coords(segment_actual_bbox, text)
                chars_and_coords = ds_utils.rectify_chars_coords(chars_and_coords)
                scaled_chars_and_coords = []
                for char_info in chars_and_coords:
                    box = char_info['box']
                    scaled_box = ds_utils.scale_bbox_int(box, x_size, y_size, base=1000)
                    scaled_chars_and_coords.append({"box": scaled_box, "text": char_info["text"]})
                for char_info in scaled_chars_and_coords:
                    bbox_string = "\t".join([str(x) for x in char_info['box']])
                    char_str = "{}\t{}".format(bbox_string, char_info['text'])
                    ret.append(char_str)
    return ret


def seg_cn_file(fn, out_fn, max_len=510):
    """
        过长就截断, 仅仅针对中文、或者简单的中英文混合的情况;
        直接按照字节来计算长度了 所以就不引入tokenizer 来估算实际长度;
        这种做法和我们java 版本tokenizer一致;
        这块潜在的有个gap， 和官方的tokenizer 是有不一样的地方的
    :param fn:
    :param out_fn:
    :return:
    """
    subword_len_counter = 0
    with open(fn, "r", encoding="utf8") as f_p, open(
            out_fn, "w", encoding="utf8"
    ) as fw_p:
        for ori_line in f_p:
            line = ori_line.rstrip()

            if not line:
                fw_p.write("\n")
                subword_len_counter = 0
                continue
            # token = line.split("\t")[0]
            # current_subwords_len = len(tokenizer.tokenize(token))
            current_subwords_len = 1

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write(ori_line + "\n")
                subword_len_counter = current_subwords_len
                continue
            subword_len_counter += current_subwords_len
            fw_p.write(ori_line)
        logging.info("write segmented data(chunk size: {}) to file {} DONE".format(max_len, out_fn))


def paddle_ocr_to_cn_dataset(dir_path, output_fn):
    """
        paddle ocr format:  文件首行是文件 x, y 大小
            从左上角开始，顺时针(x,y) 坐标一共8个字段 然后是分数 最后是内容
            3888    5184
            413.0   79.0    514.0   79.0    514.0   109.0   413.0   109.0   0.999459445476532       河南省
            510.0   75.0    778.0   79.0    777.0   116.0   510.0   112.0   0.8242834210395813      医疗伟院收费票据
            919.0   88.0    1047.0  88.0    1047.0  107.0   919.0   107.0   0.9605546593666077      票据代码：豫财4102
            232.0   103.0   420.0   103.0   420.0   126.0   232.0   126.0   0.988480269908905       郑州大学第附属医院

        layoutlm format: 左上， 右下 两个点坐标 共四个字段 需要 用1000 来进行缩放
        注意: 无数据格式验证 要保证输入符合格式要求
    :param dir_path:
    :return:
    """
    logging.info("paddle_ocr_to_cn_dataset: {} {}".format(dir_path, output_fn))
    full_fns = ds_utils.collect_files(dir_path, suffix='.txt')
    logging.info("all {} files".format(len(full_fns)))
    tmp_output_fn = output_fn + '.tmp'
    i = 0
    with open(tmp_output_fn, 'w') as fw:
        for fn in tqdm.tqdm(full_fns):
            i += 1
            # for debug only
            if i != i:
                break
            try:
                with open(fn, encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = paddle_format_to_layoutlm_format(lines, fn)
                    if len(lines) != 0:
                        fw.write("\n".join(lines) + "\n")
                fw.write("\n")
            except Exception as e:
                logging.info("ERROR in file: [{}], {}, IGNORE".format(fn, e))
    logging.info("write to tmp file: {} DONE".format(tmp_output_fn))


def pa_preprocessed_ocr_to_cn_dataset(dir_path, output_fn):
    """
         pa_preprocessed_ocr format:  每一行就是一个字
            从左上角开始，顺时针(x,y) 坐标一共8个已经缩放的坐标字段， 算上识别出来的字 最后有9个字段;
        layoutlm format: 左上， 右下 两个点坐标 共四个字段
        注意: 无数据格式验证 要保证输入符合格式要求
    :param dir_path:
    :return:
    """
    logging.info("paddle_ocr_to_cn_dataset: {} {}".format(dir_path, output_fn))
    full_fns = ds_utils.collect_files(dir_path, suffix='.txt')
    logging.info("all {} files".format(len(full_fns)))
    tmp_output_fn = output_fn + '.tmp'
    i = 0
    with open(tmp_output_fn, 'w') as fw:
        for fn in tqdm.tqdm(full_fns):
            i += 1
            # for debug only
            if i != i:
                break
            try:
                with open(fn, encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = pa_preprocessed_format_to_layoutlm_format(lines, fn)
                    if len(lines) != 0:
                        fw.write("\n".join(lines) + "\n")
                fw.write("\n")
            except Exception as e:
                logging.info("ERROR in file: [{}], {}, IGNORE".format(fn, e))
    logging.info("write to tmp file: {} DONE".format(tmp_output_fn))


if __name__ == '__main__':
    dir_path = '/home/ana/data2/datasets/layout/大赛1000训练用数据集_ocr'
    output_fn = os.path.join(kp_setup.data_dir, 'datasets', 'cn_pretrain', 'preprocess.txt')
    utils.make_sure_dir_there(os.path.dirname(output_fn))
    # paddle_ocr_to_cn_dataset(dir_path, output_fn)
    pa_preprocessed_ocr_to_cn_dataset(dir_path, output_fn)
    seg_cn_file(output_fn + '.tmp', output_fn)
