import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np

def predict(x):
    model = joblib.load('./src/iris.pkl')
    x = x.reshape(1, -1)
    pred_label = model.predict(x)
    return pred_label

def getName(label):
    if label == 0:
        return "Setosa"
    elif label == 1:
        return "Versicolor"
    elif label == 2:
        return "Virginica"
    else:
        return "Error"

app = Flask(__name__)

class IrisForm(Form):
    SepalLength = FloatField('がくの長さ(0cm 〜 10cm)',
                        [validators.InputRequired(),
                         validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')])
    
    SepalWidth = FloatField('がくの幅(0cm 〜 5cm)',
                        [validators.InputRequired(),
                         validators.NumberRange(min=0, max=5, message='0〜5の数値を入力してください')])

    PetalLength = FloatField('花弁の長さ(0cm 〜 10cm)',
                        [validators.InputRequired(),
                         validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')])
    
    PetalWidth = FloatField('花弁の幅(0cm 〜 5cm)',
                        [validators.InputRequired(),
                         validators.NumberRange(min=0, max=10, message='0〜5の数値を入力してください')])

    submit = SubmitField('判定')

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    irisForm = IrisForm(request.form)

    if request.method == 'POST':
        if irisForm.validate() == False:
            return render_template('index.html', forms=irisForm)
        else:
            VarSepalLength = float(request.form['SepalLength'])
            VarSepalWidth = float(request.form['SepalWidth'])
            VarPetalLength = float(request.form['PetalLength'])
            VarPetalWidth = float(request.form['PetalWidth'])
            x = np.array([VarSepalLength, VarSepalWidth, VarPetalLength, VarPetalWidth])
            pred = predict(x)
            irisName_ = getName(pred)
            return render_template('result.html', irisName=irisName_)
    elif request.method == 'GET':
        return render_template('index.html', forms=irisForm)
    
if __name__ == '__main__':
    app.run(debug=False)