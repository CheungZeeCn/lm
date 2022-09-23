"""
    参照 官方layoutLM 的结构 做 custom 工作
"""
from transformers import AutoTokenizer, LayoutLMModel, BertConfig, BertTokenizer, BertForMaskedLM

class CustomLayoutLMModel(LayoutLMModel):
    @classmethod
    def init_from_bert(cls, bert_model_dir, layoutlm_config_path):
