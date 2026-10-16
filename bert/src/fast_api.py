from predict_fun import predict
from fastapi import FastAPI
from fastapi import Request
import uvicorn # 异步服务器
# 1. 实例化FastAPI对象
app1 = FastAPI()
# 2. 定义路由
@app1.post("/predict")
async def predict_api(request: Request):
    # 1.定义异常
    try:
        print("接收到的数据类型:", request)
        data = await request.json()
        # 3 . 模型预测
        result = predict(data)
        return result
    except Exception as e:
        return {"error":str(e)},400
if __name__ == '__main__':
    uvicorn.run(app1, host='0.0.0.0', port=8003)