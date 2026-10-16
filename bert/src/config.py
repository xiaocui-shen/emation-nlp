import torch.cuda
from transformers import BertTokenizer
import datetime

class Config():
    def __init__(self):
        # todo 1.源文件路径
        self.train_datapath = '../../data/train.txt' #训练集路径
        self.test_datapath = '../../data/test.txt' # 测试集路径
        self.dev_datapath = '../../data/dev.txt' # 验证集路径
        self.class_path = '../../data/class.txt'  # 标签类别文件路径
        with open(self.class_path,'r',encoding='utf-8') as f:
            self.class_list = f.read().strip().split('\n') # 标签类别
        self.log_path = f'../log/log_{datetime.datetime.now().strftime("%Y%m%d")}.txt' # 日志保存路径

        # todo 2.模型保存路径
        self.best_model_save_path = '../save_model/bert_classifier_best_model.pt'
        self.last_model_save_path = '../save_model/bert_classifier_last_model.pt'
        # todo 3.bert模型相关配置
        # 模型名称,模型文件路径,隐藏层的维度
        self.model_name = 'bert'
        self.bert_path = '../bert-base-uncased'
        self.bert_hidden_size = 768
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_path) # bert分词器
        # todo 4.训练参数
        # 类别数量、训练轮次、批次大小、最大句子长度 、学习率
        self.num_classes = len(self.class_list)
        self.epochs = 3
        self.batch_size = 256
        self.max_len = 32
        self.learning_rate = 5e-5

        # todo 5. 设备配置 如果是mac本写成 'mps'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'