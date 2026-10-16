"""
该脚本的目的是: 为BERT预训练模型准备数据, 包含: 原始数据加载, 自定义数据集, 批量处理, 构架数据加载器等功能...
todo 0:加载配置文件和加载分词器
todo 1:定义函数, 实现: 加载并处理原始数据.
函数功能: 从指定文本文件中加载数据, 处理成'文本内容-标签索引'的元组列表, 供后续数据集封装使用.
    :param file_path: 原始数据文件路径. 文件内容(每行数据)格式为: '文本\t标签'
    :return:
        list: 元素为元组形式, 格式为: [('文本标题', '标签索引'), ('文本标题', '标签索引'), ...]
        例如: [('体验2D 巅峰 倚天屠龙记十大创新概览', 8), ('六月触发考研“多米诺”--统考专业课指导', 3), ...]

todo 2. 自定义数据集类.
自定义的数据集类, 封装"文本-标签"数据, 使PyTorch的DataLoader能批量加载.
我们自定义的数据集类的作用: 将原始数据(load_raw_data函数的输出结果) 转换为DataLoader可识别的格式.

todo 3. 定义批数据处理函数 collate_fn
函数作用: 最DataLoader的一个批次(batch)原始数据进行预处理, 例如: 分词, 填充, 转张量.
    :param batch: 格式为: [(标题1, 分类索引), (标题2, 分类索引), (text3, label3)...]
    :return: 包含3个PyTorch张量的元组
        input_ids: 分词后token的id, 形状为: (batch_size, max_length)
        attention_mask: 注意力掩码(标记有效token 和 填充token), 形状和 input_ids一致.
        labels: 批次标签, 形状是: (batch_size, )

todo 4. 构建数据加载器
函数作用: 构建训练集, 测试集, 验证集的数据加载器, 供模型训练, 评估使用.
    :return: 包含3个DataLoader的元组, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)


"""

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import Config
from transformers import BertTokenizer

conf = Config()
tokennizer = BertTokenizer.from_pretrained(conf.bert_path)


def load_row_data(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loadding data'):
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            label = int(label)
            data.append((text, label))
    return data


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, y = self.data[item]
        return x, y


def collate_fn(batch):
    # 提取文本标签
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    text_tokens = tokennizer(texts,  # 要编码的文本
                                               add_special_tokens=True,  # 是否添加特殊标记 （cls，sep）
                                               padding="max_length",  # 填充策略
                                               max_length=conf.max_len,  # 最大句子长度
                                               truncation=True,  # 是否截断
                                               return_attention_mask=True  # 返回注意力掩码(区分有效token和填充token)
                                               )
    input_ids = torch.tensor(text_tokens["input_ids"])
    attention_mask = torch.tensor(text_tokens["attention_mask"])
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


def build_dataloader():
    # 加载原始数据
    train_data = load_row_data(conf.train_datapath)
    test_data = load_row_data(conf.test_datapath)
    dev_data = load_row_data(conf.dev_datapath)
    # 创建dataset
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)
    # 创建dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    for i in train_dataloader:
        print(i)
        break
