from config import  Config
from bilstm_classifier import BiLSTMClassifier
import torch

conf = Config()
# 1. 加载模型
model = BiLSTMClassifier().to(conf.device)
model.load_state_dict(torch.load(conf.student_last_model_save_path))
# 2.将模型设置为评估模式
model.eval()
# 3.预测
def predict(data):
    # 1. 分词
    text = data["text"]
    token_ids = conf.bert_tokenizer(text, return_tensors="pt").to(conf.device)
    print(token_ids)
    with torch.no_grad():
        input_ids = token_ids.input_ids.to(conf.device)
        attention_mask = token_ids.attention_mask.to(conf.device)
        logits = model(input_ids, attention_mask)
        # 拿到分类的索引号
        y_pred = torch.argmax(logits, dim=1).item()
        # 拿到分类的名称
        label_name  = conf.class_list[y_pred]
        data["pred_class"] = label_name
        return data

if __name__ == '__main__':
    print(predict({'text':"名师辅导：2012考研英语虚拟语气三种用法"}))