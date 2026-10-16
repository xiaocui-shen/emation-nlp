"""
# 该脚本用于: BERT分类模型的定义 和 创建.
todo 1.创建配置文件对象.
todo 2. 定义BERT分类模型(继承nn.Module), 实现自定义神经网络.
"""
import torch.nn as nn
import torch
from transformers import BertModel
from config import Config

# todo 1.创建配置文件对象.
config = Config()


# todo 2. 定义BERT分类模型(继承nn.Module), 实现自定义神经网络.
class Bert_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 2.构建输出层 输入是bert的隐藏层大小 输出是类别数量
        self.fc = nn.Linear(config.bert_hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        # output : last_hidden_state  pooler_output

        out = self.fc(output.pooler_output)
        return out


if __name__ == '__main__':
    from utils import build_dataloader

    model = Bert_Classifier()
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    for input_ids, attention_mask, labels in train_dataloader:
        output = model(input_ids, attention_mask)
        print(output.shape)
        break
