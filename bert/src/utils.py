from config import Config
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch

# todo 0:加载配置文件
conf = Config()

# todo 1:定义函数，实现：加载并处理原始数据。
def load_row_data(file_path):
    """
    函数功能：从指定文本文件中加载数据，处理成“文本内容-标签索引”的元组列表，供后续数据集封装使用。
    :param file_path: 原始数据文件路径，文件内容(每行数据)格式为：'文本\t标签'
    :return:
        list: 元素为元组形式，格式为：[('文本标题', '标签索引'), ('文本标题', '标签索引'), ...]
        例如：[('体验2D 巅峰 倚天屠龙记十大创新概览', 8), ('六月触发考研“多米诺”一统考专业课指导', 3), ...]
    """
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in tqdm(f,desc='Loading data'):
            line = line.strip()
            # 去掉空值
            if not line:
                continue
            # 分割文本和标签
            text, label = line.split('\t')
            data.append((text, int(label)))
    return data

# todo 2. 自定义数据集类
class TextDataset(Dataset):
    """
    自定义的数据集类,封装"文本-标签"数据,使PyTorch的DataLoader能批量加载.
    我们自定义数据集类的作用:将原始数据(load_raw_data函数的输出结果)转换为DataLoader可识别的格式.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x,y = self.data[item]
        return x,y
# todo 3. 定义批数据处理函数 collate_fn
def collate_fn(batch):
    """
    函数作用：对DataLoader的一个批次(batch)原始数据进行预处理。例如：分词、填充、转张量。
    :param batch: 格式为：[((标题1, 分类索引), (标题2, 分类索引), (text3, label3)...)]
    :return: 包含3个PyTorch张量的元组
              input_ids: 分词后token的id, 形状为: (batch_size, max_length)
              attention_mask: 注意力掩码(标记有效token 和 填充token), 形状和 input_ids一致
              labels: 批次标签, 形状是: (batch_size, )
    """
    # 1.获取文本和标签
    texts = [item[0] for item in batch]
    label = [item[1] for item in batch]
    # 2.将文本转token
    tokenizer_output = conf.bert_tokenizer(texts, # 输入文本
                                           truncation=True, # 截断
                                           padding='max_length', # 填充
                                           max_length=conf.max_len)# 最大长度
    # 3.将索引转张量
    input_ids = torch.tensor(tokenizer_output['input_ids']) # 输入_ids
    attention_mask = torch.tensor(tokenizer_output['attention_mask'])
    labels = torch.tensor(label)
    return input_ids, attention_mask, labels

# todo 4. 构建数据加载器
def build_data_loader():
    """
    函数作用：构建训练集，测试集，验证集的数据加载器，供模型训练、评估使用。
    :return: 包含3个DataLoader的元组，顺序为：(train_dataloader, test_dataloader, dev_dataloader)
    """
    # 加载数据
    train_data = load_row_data(conf.train_datapath)
    test_data = load_row_data(conf.test_datapath)
    dev_data = load_row_data(conf.dev_datapath)
    # 构建数据集对象
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)
    # 构建数据加载器 为每一个批次函数进行数据预处理
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, test_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader, dev_dataloader = build_data_loader()
    for input_ids, attention_mask, labels in train_dataloader:
        print(input_ids.shape, attention_mask.shape, labels.shape)
        break



