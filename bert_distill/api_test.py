import requests
import time
# 定义预测接口地址
url = 'http://127.0.0.1:8004/predict'
# 构造请求数据

data = {'text': "中华女子学院：本科层次仅1专业招男生"}
# 记录开始时间
start_time = time.time()
# 发送 POST 请求
response = requests.post(url, json=data)
# 计算耗时（毫秒）
elapsed_time = (time.time() - start_time) * 1000
print(f"请求耗时: {elapsed_time:.2f} ms")
result = response.json()
print(f"预测结果: {result['pred_class']}")
