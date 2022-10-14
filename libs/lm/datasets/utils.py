import logging
import os


def collect_files(dir_path, suffix='.txt'):
    full_fns = []
    for path_info in os.walk(dir_path):
        if len(path_info[1]) == 0 and len(path_info[2]) != 0:
            for fn in path_info[2]:
                if fn.endswith(suffix):
                    full_fn = os.path.join(dir_path, path_info[0], fn)
                    full_fns.append(full_fn)
    return full_fns


def get_char_size(ch):
    return 1


def scale_bbox_int(box, width, length, base=1000):
    return [
        min(base-1, int(base * (box[0] / width))),
        min(base-1, int(base * (box[1] / length))),
        min(base-1, int(base * (box[2] / width))),
        min(base-1, int(base * (box[3] / length)))
    ]


def rectify_chars_coords(chars_and_coords):
    result = []
    for ch_info in chars_and_coords:
        x0, y0, x1, y1 = ch_info['box']
        if x1-x0 < 0:
            logging.warning("x1-x0 < 0: rectify_chars_coords check: {}".format(ch_info))
            ch_info['box'][2] = ch_info['box'][0]
        if y1-y0 < 0:
            logging.warning("y1-y0 < 0: rectify_chars_coords check: {}".format(ch_info))
            ch_info['box'][3] = ch_info['box'][1]
        result.append(ch_info)
    return result


def cn_segment_coord_to_chars_coords(segment_actual_bbox, text):
    """
        仅仅支持横排文字
    :param segment_actual_bbox: 四个 float 的列表
    :param text: 对应的文本
    :return:
    """
    offsets = []
    sizes = []
    seq_len = 0
    chars_coords = []
    for i, ch in enumerate(text):
        i_size = get_char_size(ch)
        offsets.append(seq_len)
        sizes.append(i_size)
        seq_len += i_size

    x_size = segment_actual_bbox[2] - segment_actual_bbox[0]
    y_size = segment_actual_bbox[3] - segment_actual_bbox[1]

    for i, ch in enumerate(text):
        x0 = segment_actual_bbox[0] + (offsets[i] / seq_len) * x_size
        y0 = segment_actual_bbox[1]
        x1 = segment_actual_bbox[2] + (offsets[i] + sizes[i]) / seq_len * x_size
        y1 = segment_actual_bbox[3]

        ch_info = {"box": [x0, y0, x1, y1], "text": ch}
        chars_coords.append(ch_info)

    return chars_coords
