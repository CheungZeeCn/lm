"""
    参照 官方layoutLM 的结构 做 custom 工作
"""
import logging
from torch import nn
from transformers import LayoutLMConfig, LayoutLMModel, BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
    LayoutLMForMaskedLM, LayoutLMForTokenClassification
import kp_setup


class CustomLayoutLMModel(LayoutLMForMaskedLM):
    @classmethod
    def init_from_bert(cls, bert_model_dir, layoutlm_config_path):
        logging.info("init_from_bert")
        # bert
        bert_model = BertForMaskedLM.from_pretrained(bert_model_dir)

        layoutlm_config = LayoutLMConfig.from_json_file(layoutlm_config_path)
        logging.info(layoutlm_config)
        layout_model = LayoutLMForMaskedLM(layoutlm_config)

        def copy_bert_weights_to_layoutlm(bert_model, layout_model):
            # copyt emb:
            # copy encoder:
            # copy cls:
            def copy_block_to(from_block, to_block):
                # to_block_params = {k:v for k, v in to_block.named_parameters()}
                to_block_params = dict(to_block.named_parameters())

                for name, param in from_block.named_parameters():
                    if name in to_block_params:
                        to_block_params[name].data.copy_(param.data)

            copy_block_to(bert_model.bert.embeddings, layout_model.layoutlm.embeddings)

        copy_bert_weights_to_layoutlm(bert_model, layout_model)
        logging.info("init from {} DONE".format(bert_model_dir))
        return layout_model




if __name__ == '__main__':
    bert_model_dir = '/home/ana/data2/models/chinese-roberta-wwm-ext'
    layoutlm_config_path = '/home/ana/data2/models/chinese-roberta-wwm-ext-layoutlm/config.json'
    layoutlm_model = CustomLayoutLMModel.init_from_bert(bert_model_dir, layoutlm_config_path)
