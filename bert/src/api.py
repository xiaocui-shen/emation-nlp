from predict_fun import predict
from flask import Flask,request

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return {'error': '请提供有效的文本数据'}, 400
        result = predict(data)
        print(result)
        return result

    except Exception as e:
        return {'error': str(e)}, 500
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8003)