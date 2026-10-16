from predict_fun import predict
from flask import Flask,request

app = Flask(__name__)

@app.route('/predict',methods=["post"])
def predict_api():
    try:
        data = request.get_json()
        result = predict(data)
        print(result)
        return result

    except Exception as e:
        return {'error':str(e)}
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8004)