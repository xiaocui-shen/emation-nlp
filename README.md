# 情智星球 · Emation

英文语句情感六分类（anger / fear / joy / love / sadness / surprise）—— 以 `bert-base-uncased` 微调为教师模型，再通过知识蒸馏压缩到 BiLSTM 学生模型，配套 Flask / FastAPI 推理服务与 Streamlit 交互前端。

## 项目简介

| 能力 | 说明 |
| --- | --- |
| 迁移学习 | 加载 `bert-base-uncased` 预训练权重，接一层全连接做 6 分类微调（`bert/`） |
| 知识蒸馏 | 用教师模型的软标签 + 真实硬标签联合训练 BiLSTM 学生模型（`bert_distill/`） |
| 在线服务 | Flask / FastAPI 两种 `POST /predict` 实现，返回情感类别 |
| 交互前端 | Streamlit 页面：中文输入 → DeepSeek 翻译为英文 → 调模型预测 → 结合情感结果与 AI 助手对话 |

## 模型效果（实测）

数据来自本仓库训练日志（`bert/log/`、`bert_distill/log/`），测试集 58,353 条。

| 模型 | 结构 | 权重体积 | 测试集 Acc | 测试集 weighted-F1 |
| --- | --- | --- | --- | --- |
| BERT 教师 | bert-base-uncased + Linear(768→6) | 418 MB | 0.92 | 0.92 |
| BiLSTM 学生（蒸馏后） | Embedding 512 + BiLSTM(512×3) + Dropout 0.2 + Linear | 124 MB | 0.9069 | 0.9074 |

结论：学生模型体积约为教师的 **30%**，测试集精度仅下降约 **1.3 个百分点**，蒸馏后仍保持 0.91 级别表现。

类别不均衡是主要误差来源 —— 样本较少的 `love`(34,554) 与 `surprise`(14,972) 两类，精确率/召回率明显失衡（如教师模型 `love` 精确率 0.99 但召回 0.68，`surprise` 精确率 0.74 但召回 0.96）。

## 数据集

- 语料：英文 Twitter 短文本，单标签 6 分类，原始文件 `data/original_data.txt`（含表头 `text\temotions`，共 416,808 条有效样本）。
- 标签顺序由 `data/class.txt` 定义：`anger=0, fear=1, joy=2, love=3, sadness=4, surprise=5`。
- 原始标签分布：joy 141,067 / sadness 121,187 / anger 57,317 / fear 47,712 / love 34,554 / surprise 14,972。
- 切分后实际用于训练：共 291,766 条（train 204,236 / dev 29,177 / test 58,353，约 7 : 1 : 2），格式为每行 `文本\t标签索引`。

| 类别索引 | 标签 | train | dev | test |
| --- | --- | --- | --- | --- |
| 0 | anger | 28,226 | 3,950 | 8,090 |
| 1 | fear | 23,498 | 3,363 | 6,664 |
| 2 | joy | 69,194 | 9,927 | 19,675 |
| 3 | love | 17,009 | 2,411 | 4,807 |
| 4 | sadness | 59,051 | 8,506 | 16,970 |
| 5 | surprise | 7,258 | 1,020 | 2,147 |

## 目录结构

```
emation/
├── bert/                        # 教师模型：BERT 微调
│   ├── src/
│   │   ├── config.py            # 配置（路径 / 超参数 / 设备）
│   │   ├── bert_classifier.py   # BERT + Linear 分类模型定义
│   │   ├── utils.py             # 数据加载、Dataset、collate_fn、DataLoader
│   │   ├── train.py             # 训练 + 验证 + 保存最优模型
│   │   ├── predict_fun.py       # 单条文本预测（加载 best 权重）
│   │   ├── api.py               # Flask 接口（:8003）
│   │   ├── fast_api.py          # FastAPI 接口（:8003）
│   │   ├── api_test.py          # 接口调用测试
│   │   ├── app_streamlit.py     # Streamlit 前端（含 DeepSeek 翻译 + 对话）
│   │   └── test.html            # 产品介绍静态页
│   ├── save_model/              # 权重产物（未提交，见 .gitignore）
│   └── log/                     # 训练日志
├── bert_distill/                # 学生模型：知识蒸馏
│   ├── config.py                # 路径 / BiLSTM 参数 / 蒸馏温度与权重
│   ├── teacher_model.py         # 教师模型（与 bert 结构一致，用于加载权重）
│   ├── bilstm_classifier.py     # BiLSTM 学生模型
│   ├── utils.py                 # 数据管道（与 bert 版本一致）
│   ├── bert_distill_train.py    # 蒸馏训练主流程
│   ├── predict_fun.py           # 学生模型预测
│   ├── api.py                   # Flask 接口（:8004）
│   ├── api_test.py              # 接口调用测试
│   ├── save_model/              # 权重产物（未提交）
│   └── log/                     # 训练日志
└── data/
    ├── original_data.txt        # 原始全量数据（text \t emotions）
    ├── merged_training.pkl      # 合并后的全量 DataFrame
    ├── train.txt / dev.txt / test.txt
    ├── class.txt                # 标签顺序定义
    └── stopwords.txt            # 停用词表（当前代码未使用）
```

## 环境依赖

未固定版本，推荐 Python 3.10 + CUDA 环境：

```bash
pip install torch transformers scikit-learn tqdm
pip install flask fastapi uvicorn requests
pip install streamlit openai
```

> 权重文件（`*.pt`）与 `bert-base-uncased` 预训练模型均已在 `.gitignore` 中排除，克隆后需自行准备。

## 快速开始

**0. 准备预训练模型**

将 `bert-base-uncased` 放到 `bert/bert-base-uncased/`（目录名与 `config.py` 中 `bert_path` 一致）。

**1. 训练教师模型**

```bash
cd bert/src
python train.py
```

产物：`bert/save_model/bert_classifier_best_model.pt`（按验证集 F1 最优保存）与 `bert_classifier_last_model.pt`。

**2. 蒸馏学生模型**

```bash
cd bert_distill
python bert_distill_train.py
```

教师权重读取路径为 `../bert/save_model/bert_classifier_best_model.pt`，请先完成第 1 步。产物：`bert_distill/save_model/student_classifier_best_model.pt` 与 `student_classifier_last_model.pt`。

**3. 启动推理服务**（二选一）

```bash
cd bert/src
python api.py          # Flask  :8003
python fast_api.py     # FastAPI :8003
```

学生模型服务（蒸馏版）：

```bash
cd bert_distill
python api.py          # Flask :8004
```

**4. 启动前端**

```bash
# 先设置 DeepSeek API Key（用于中文→英文翻译与情感对话）
export DEEPSEEK_API_KEY=sk-xxxx      # Windows: set DEEPSEEK_API_KEY=sk-xxxx

cd bert/src
streamlit run app_streamlit.py
```

前端流程：输入中文 → DeepSeek 翻译成英文 → 请求 `http://localhost:8003/predict` → 展示情感类别（中文）→ 进入基于该情感的 AI 助手对话。

**5. 调用示例**

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8003/predict",
    json={"text": "i feel amazing today"},
)
print(resp.json())   # {'text': 'i feel amazing today', 'pred_class': 'joy'}
```

## 接口说明

| 项 | 值 |
| --- | --- |
| 方法 | `POST` |
| 路径 | `/predict` |
| 端口 | `8003`（`bert/src`）、`8004`（`bert_distill`） |
| 请求体 | `{"text": "english sentence"}` |
| 成功响应 | `{"text": "...", "pred_class": "joy"}` |
| 失败响应 | `{"error": "错误信息"}` |

## 核心实现

**教师模型**：`BertModel.from_pretrained(bert_path)` 取 `pooler_output`（768 维），接 `Linear(768, 6)` 输出 logits；损失 `CrossEntropyLoss`，优化器 `AdamW`。

**学生模型**：`Embedding` → 用 `attention_mask` 将 padding 位置置零 → 3 层双向 LSTM → 取每句最后一个时间步输出 → `Dropout` → `Linear(1024, 6)`。

**蒸馏损失**：

```
soft_loss = KLDivLoss(student_logits / T, teacher_logits / T)
hard_loss = CrossEntropyLoss(student_logits, labels)
loss = alpha * soft_loss * T^2 + (1 - alpha) * hard_loss
```

其中 `T = 3.0`（温度，乘 `T²` 使软/硬损失量级对齐），`alpha = 0.5`（软标签权重）。教师模型只做前向（`eval()` + `torch.no_grad()`），仅更新学生模型参数。

**训练超参数**（`config.py`）：

| 阶段 | epochs | batch_size | max_len | lr | 其他 |
| --- | --- | --- | --- | --- | --- |
| BERT 教师 | 3 | 256 | 32 | 5e-5 | 每 100 步在验证集评估并按 F1 保存最优 |
| BiLSTM 学生 | 5 | 512 | 32 | 1e-4 | T=3.0, alpha=0.5，每 240 步按 F1 保存最优 |

## 运行注意

- 所有脚本使用**相对路径**（如 `../../data/train.txt`、`../bert-base-uncased`），必须在各自目录（`bert/src`、`bert_distill`）下执行。
- 模型基于英文语料训练，中文输入需先翻译（前端已用 DeepSeek 处理）。
- `bert_distill/predict_fun.py` 加载的是 `student_classifier_last_model.pt`（最后一个 epoch 权重），而非 best 权重。
- `predict_fun.py` 中 `torch.load` 未指定 `map_location`，在无 GPU 机器上加载 CUDA 权重可能报错。

## 数据来源与致谢

- 数据集：英文 Twitter 情感数据集（六类基本情感），原始分布与本项目 `original_data.txt` 完全一致。
- 论文：Saravia et al. *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018.
- 预训练模型：`google-bert/bert-base-uncased`。
- 前端对话能力：DeepSeek API。
