"""
该脚本用于: BERT版投满分案例的 模型构建文件
"""


from config import Config
import torch.nn as nn
conf = Config()

class BiLSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        # 1.词嵌入层
        self.embed = nn.Embedding(conf.vocab_size, conf.embedding_size)
        # 2.双向lstm层
        self.lstm = nn.LSTM(conf.embedding_size, # 输入维度大小
                            conf.lstm_hidden_size, # 隐藏层维度大小
                            conf.lstm_layers, # 隐藏层层数
                            bidirectional=conf.is_bilstm ,# 是否使用双向lstm
                            batch_first=True) # 是否使用batch_first模式
        # 3.fc层
        self.fc = nn.Linear(conf.lstm_hidden_size * 2, conf.num_classes)
        # 4.dropout层
        self.dropout = nn.Dropout(conf.dropout)

    def forward(self,input_ids,attention_mask):
        # 1.数据进入词嵌入层 input_ids:[batch,max_len]-->[batch,max_len,embedding_size]
        #     attention_mask:[batch,max_len]
        embed = self.embed(input_ids)
        # 2.讲填充的token置为0 [batch,max_len,-1]
        attention_mask = attention_mask.unsqueeze(-1)
        embed = embed*attention_mask
        # 3.进入lstm层 [batch,max_len,embedding_size]-->[batch,max_len,lstm_hidden_size*2]
                                                      # 128x32x1024 -->128x1024
        lstm_out, (h, c) = self.lstm(embed)
        # 4.取每一个句子的最后一个单词的输出 [batch,lstm_hidden_size*2]
        output = lstm_out[:,-1]
        # 5.数据进入dropout层
        output = self.dropout(output)
        # 6.数据进入fc层 [batch,lstm_hidden_size*2]-->[batch,num_classes]
        logits = self.fc(output)
        return logits
if __name__ == '__main__':
    from utils import build_dataloader
    _,test_dataloader,_ = build_dataloader()
    model = BiLSTMClassifier()
    for batch in test_dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        output = model(input_ids,attention_mask)
        print(output.shape)
        break