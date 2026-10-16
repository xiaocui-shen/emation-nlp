import torch
from config import Config
from utils import build_data_loader
from bert_classifier import Bert_Classifier
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

# todo 1.加载配置文件 和 核心工具类
conf = Config()
# todo 2.定义模型训练函数: 封装完成的训练流程,例如：数据加载 -> 模型训练 -> 验证 -> 保存.
def model2train(train_dataloader, dev_dataloader, model):
    # 1.损失函数和优化器
    loss_f = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    # 2.遍历模型训练轮次
    total_item = 0 # 迭代次数
    best_f1 = 0 # 最好的f1值
    for epoch in range(conf.epochs):
        # 将模型设置为训练模式
        model.train()
        # 定义训练监控日志变量
        total_loss,y_pre_list,y_true_list,start_time = 0.,[],[],time.time() #损失值、预测值、真实值、开始时间
        # 遍历训练数据
        for i, batch in tqdm(enumerate(train_dataloader),desc=f'模型训练中,Epoch{epoch+1}'):
            input_ids = batch[0].to(conf.device) # 输入_ids
            attention_mask = batch[1].to(conf.device) # attention_mask
            labels = batch[2].to(conf.device) # 标签
            # 1.前向传播
            outputs = model(input_ids, attention_mask)
            # 2.计算损失
            loss = loss_f(outputs, labels)
            # 3.梯度清零
            optimizer.zero_grad()
            # 4.反向传播
            loss.backward()
            # 5.参数更新
            optimizer.step()
            # 6.记录训练监控日志
            total_loss += loss.item() # 累加损失值
            y_pre = torch.argmax(outputs, dim=1) # 预测值
            y_pre_list.extend(y_pre.cpu().tolist()) # 添加到预测值列表
            y_true_list.extend(labels.cpu().tolist()) # 添加到真实值列表
            total_item += 1 # 累加迭代次数
            # 7.每30步或者批次末尾打印一次模型训练日志
            if total_item % 30 == 0 or total_item == len(train_dataloader):
                acc = accuracy_score(y_true_list, y_pre_list)
                f1 = f1_score(y_true_list, y_pre_list, average='weighted')
                # 打印模型训练日志 训练轮次 步数 损失值 准确率 f1值 时间
                print(f'Epoch{epoch+1} Step{total_item} Loss{total_loss/total_item} Acc{acc} F1{f1} Time{time.time()-start_time}')
                # 保存训练日志
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(f'\nEpoch{epoch+1} Step{total_item} Loss{total_loss/total_item} Acc{acc} F1{f1} Time{time.time()-start_time}\n')
                # 训练日志清零
                total_loss,y_pre_list,y_true_list = 0.,[],[]
            # 8.每100步或者最后一步保存最优模型
            if total_item % 100 == 0 or total_item == len(train_dataloader):
                acc,precision,recall,f1,report = model2dev(dev_dataloader, model,conf.device)
                print(f'验证集评估报告:\n{report}')
                print(f'验证集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}')
                # 保存最优模型
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), conf.best_model_save_path)
                # 保存训练日志
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(f'\n验证集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}\n')
                    f.write(f'\n 验证集评估报告: \n {report}\n')
                # 保存模型
            torch.save(model.state_dict(), conf.last_model_save_path)

# todo 3. 定义模型验证函数
def model2dev(dev_dataloader,model,device):
    # 1.模型转为验证格式 正则化项不生效
    model.eval()
    # 2.定义评估指标变量
    y_pre_list,y_true_list = [],[]
    # 3.遍历验证数据
    for i,batch in tqdm(enumerate(dev_dataloader),desc=f'模型验证中...'):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        # 前向传播
        logits = model(input_ids, attention_mask)
        y_pre = torch.argmax(logits, dim=-1)
        y_pre_list.extend(y_pre.cpu().tolist())
        y_true_list.extend(labels.cpu().tolist())

    # 4.计算评估指标
    acc = accuracy_score(y_true_list, y_pre_list)
    precision = precision_score(y_true_list, y_pre_list, average='weighted')
    recall = recall_score(y_true_list, y_pre_list, average='weighted')
    f1 = f1_score(y_true_list, y_pre_list, average='weighted')
    report = classification_report(y_true_list, y_pre_list)
    return acc, precision, recall, f1, report

# todo 4. 定义主函数
def main():
    # 1. 加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_data_loader()
    # 2. 实例化模型
    model = Bert_Classifier().to(conf.device)
    # 3.模型训练
    model2train(train_dataloader, dev_dataloader, model)
    # 4.模型评估
    acc, precision, recall, f1, report = model2dev(test_dataloader, model, conf.device)
    print(f'测试集评估报告: \n {report}')
    print(f'测试集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}')
    # 保存测试集评估结果
    with open(conf.log_path, 'a', encoding='utf-8') as f:
        f.write(f'测试集评估报告: \n {report}\n')
        f.write(f'测试集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}\n')
    # pass

if __name__ == '__main__':
    main()
