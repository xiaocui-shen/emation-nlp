import torch
from transformers import  BertTokenizer
from bert_classifier import  Bert_Classifier
from config import  Config
conf = Config()
tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
model = Bert_Classifier().to(conf.device)
model.load_state_dict(torch.load('../save_model/bert_classifier_best_model.pt'))
model.eval()

def predict(data):
    #1.获取数据
    text = data['text']
    #2.预处理数据
    tokenize = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=conf.max_len)
    input_ids = tokenize['input_ids'].to(conf.device)
    attention_mask = tokenize["attention_mask"].to(conf.device)
    with torch.no_grad():
        logits = model(input_ids,attention_mask)
        pred_ids = torch.argmax(logits,dim=-1)
        pred_class = conf.class_list[pred_ids]
    data['pred_class'] = pred_class
    return data
if __name__ == '__main__':
    pre = predict({'text':'这个电影真的太差了，我完全不能给它一个差评'})
    print(pre)
