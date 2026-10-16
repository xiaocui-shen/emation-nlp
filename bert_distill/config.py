"""
该脚本用于: BERT版投满分案例的 模型蒸馏配置文件.
# todo 1.源文件路径
# 训练集路径、测试集路径、验证集路径、标签类别文件路径、标签类别、日志保存路径
#
# todo 2.老师/学生模型路径
#
# todo 3.bert模型相关配置
# 模型名称、模型文件路径、隐藏层的维度
#
# todo 4.训练参数
# 类别数量、训练轮次、批次大小、最大句子长度 、学习率
#
# todo 5.bilstm模型参数设置
# embedding维度、lstm隐藏层维度、lstm层数、dropout概率
#
# todo 6.蒸馏参数设置
# 蒸馏温度、蒸馏权重

"""
import datetime
import torch.cuda
from transformers import BertTokenizer

class Config():
    def __init__(self):

        # todo 1.源文件路径
        # 训练集路径、测试集路径、验证集路径、标签类别文件路径、标签类别、日志保存路径
        self.train_datapath = '../data/train.txt'
        self.test_datapath = '../data/test.txt'
        self.dev_datapath = '../data/dev.txt'
        self.class_path = '../data/class.txt'
        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.class_list = f.read().strip().split('\n')
        self.log_file = './log/log_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.txt'

        # todo 2.老师/学生模型路径
        self.teacher_model_path = '../bert/save_model/bert_classifier_best_model.pt'
        self.student_best_model_save_path = './save_model/student_classifier_best_model.pt'
        self.student_last_model_save_path = './save_model/student_classifier_last_model.pt'
        # todo 3.bert模型相关配置
        # 模型名称、模型文件路径、隐藏层的维度
        self.model_name = 'bert'
        self.bert_path = '../bert/bert-base-uncased'
        self.bert_hidden_size = 768
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # bert分词器
        self.vocab_size = self.bert_tokenizer.vocab_size # 词表大小
        # todo 4.老师模型训练参数
        # 类别数量、训练轮次、批次大小、最大句子长度 、学习率
        self.num_classes = len(self.class_list)
        self.epochs = 5
        self.batch_size = 512
        self.max_len = 32
        self.learning_rate = 1e-4
        # todo 5.bilstm模型参数设置
        # embedding维度、lstm隐藏层维度、lstm层数、dropout概率
        self.embedding_size = 512
        self.lstm_hidden_size = 512
        self.lstm_layers = 3
        self.dropout = 0.2
        self.is_bilstm = True  # 是否使用bilstm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # todo 6.蒸馏参数设置
        # 蒸馏温度、蒸馏权重
        self.temperature = 3.0
        self.alpha = 0.5