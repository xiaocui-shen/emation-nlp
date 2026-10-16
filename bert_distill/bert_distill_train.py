"""
模型蒸馏训练文件
# todo 1.加载配置文件
# todo 2.定义模型蒸馏函数
# todo 3.定义验证函数
# todo 4.主函数


"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from teacher_model import Bert_Classifier
from bilstm_classifier import BiLSTMClassifier
from config import Config
from utils import build_dataloader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, KLDivLoss
import time
import os

# todo 1.加载配置文件
conf = Config()


# todo 2.定义模型蒸馏函数
def bert_distill(train_dataloader, dev_dataloader, teacher_model, student_model):
    # 1. 损失函数和优化器
    soft_loss_f = KLDivLoss(reduction='batchmean')
    hard_loss_f = CrossEntropyLoss()
    optimizer = AdamW(student_model.parameters(), lr=conf.learning_rate)
    # 2. 遍历模型训练轮次
    best_f1 = 0  # 最佳F1值
    total_item = 0  # 迭代次数
    # 3. 定义T和alpha
    T = conf.temperature
    alpha = conf.alpha
    for epoch in range(conf.epochs):
        # 将学生模型转为训练格式
        student_model.train()
        # 将老师模型转为验证格式
        teacher_model.eval()
        total_loss, y_pre_list, y_true_list, start_time = 0, [], [], time.time()  # 损失值, 预测值, 真实值
        loss_item = 0
        # 遍历训练数据
        for i, batch in tqdm(enumerate(train_dataloader), desc=f'模型训练中. Epoch {epoch + 1}'):
            input_ids = batch[0].to(conf.device)
            attention_mask = batch[1].to(conf.device)
            labels = batch[2].to(conf.device)
            # 1. 前向传播
            # 1.1 学生模型前向传播
            student_logits = student_model(input_ids, attention_mask)
            # 1.2 老师模型前向传播
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)

            # 2.计算损失
            # 3.1 计算软标签损失
            soft_loss = soft_loss_f(torch.softmax(student_logits / T, dim=-1),
                                    torch.softmax(teacher_logits / T, dim=-1))
            # 3.2 计算硬标签损失
            hard_loss = hard_loss_f(student_logits, labels)

            # 3.3 计算总损失 乘T的平方是为了让soft_loss的值和hard_loss的值在同一个量级上
            loss = alpha * soft_loss * (T * T) + (1 - alpha) * hard_loss

            # 3.梯度清零
            optimizer.zero_grad()
            # 4.反向传播
            loss.backward()
            # 5. 参数更新
            optimizer.step()

            # 6. 记录训练监控日志
            total_loss += abs(loss.item())
            y_pre = torch.argmax(student_logits, dim=-1)
            y_pre_list.extend(y_pre.cpu().tolist())
            y_true_list.extend(labels.cpu().tolist())
            total_item += 1
            loss_item += 1

            # 7. 每60步或者最后一步打印一次训练监控日志
            if total_item % 60 == 0 or total_item == len(train_dataloader):
                acc = accuracy_score(y_true_list, y_pre_list)
                f1 = f1_score(y_true_list, y_pre_list, average='weighted')
                # 打印训练监控日志 训练轮次, 步数, 损失值, 精度, F1值, 时间
                print(
                    f'Epoch {epoch + 1}, Step {total_item}, Loss: {total_loss / loss_item}, Acc: {acc}, F1: {f1}, Time: {time.time() - start_time}')
                # 保存训练日志
                with open(conf.log_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f'\nEpoch {epoch + 1}, Step {total_item}, Loss: {total_loss / loss_item}, Acc: {acc}, F1: {f1}, Time: {time.time() - start_time}\n')
                # 训练日志清零
                total_loss, y_pre_list, y_true_list, loss_item = 0, [], [], 0
            # 8. 每240步或者最后一步保存最优模型
            if total_item % 240 == 0 or total_item == len(train_dataloader):
                acc, precision, recall, f1, report = model2dev(dev_dataloader, student_model, conf.device)
                print(f'验证集评估报告: \n {report}')
                print(f'验证集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}')
                # 保存最优模型
                if f1 > best_f1:
                    best_f1 = f1
                    # 确保保存目录存在
                    save_dir = os.path.dirname(conf.student_best_model_save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(student_model.state_dict(), conf.student_best_model_save_path)
                # 保存训练日志
                with open(conf.log_file, 'a', encoding='utf-8') as f:
                    f.write(f'\n验证集准确率: {acc}, 精度: {precision}, 召回率: {recall}, F1值: {f1}\n')
                    f.write(f'\n 验证集评估报告: \n {report}\n')
                student_model.train()
    # 保存模型
    # 确保保存目录存在
    save_dir = os.path.dirname(conf.student_last_model_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(student_model.state_dict(), conf.student_last_model_save_path)


# todo 3.定义验证函数
def model2dev(dev_dataloader, model, device):
    # 1. 模型转为验证格式 正则化项不生效
    model.eval()
    # 2.定义评估指标变量
    y_pre_list, y_true_list = [], []
    # 3.遍历验证数据
    for i, batch in tqdm(enumerate(dev_dataloader), desc=f'模型验证中.'):
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


# todo 4.主函数
def main():
    # 1.加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 2.加载模型
    # 2.1加载老师模型
    teacher_model = Bert_Classifier()
    teacher_model.load_state_dict(torch.load(conf.teacher_model_path, map_location=torch.device('cpu')))
    teacher_model.to(conf.device)
    # 2.1 实例化学生模型
    student_model = BiLSTMClassifier().to(conf.device)
    # 3.模型蒸馏
    bert_distill(train_dataloader, dev_dataloader, teacher_model, student_model)
    # 4.模型测试
    acc, precision, recall, f1, report = model2dev(test_dataloader, student_model, conf.device)
    # 5.打印测试指标和保存测试日志
    print(f'acc：{acc}')
    print(f'precision：{precision}')
    print(f'recall：{recall}')
    print(f'f1：{f1}')
    print(f'report：\n{report}')
    # 保存测试日志
    with open(conf.log_file, 'a', encoding='utf-8') as f:
        f.write(f'acc：{acc}\n')
        f.write(f'precision：{precision}\n')
        f.write(f'recall：{recall}\n')
        f.write(f'f1：{f1}\n')
        f.write(f'report：\n{report}\n')
if __name__ == '__main__':
    main()