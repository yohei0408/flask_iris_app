# 必要なモジュールのインポート
import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np

# 学習済みモデルをもとに推論する関数
def predict(x):
    # 学習済みモデル（iris.pkl）を読み込み
    model = joblib.load('./src/iris.pkl')
    x = x.reshape(1,-1)
    pred_label = model.predict(x)
    return pred_label

# 推論したラベルから花の名前を返す関数
def getName(label):
    if label == 0:
        return 'Iris Setosa'
    elif label == 1:
        return 'Iris Versicolor'
    elif label == 2:
        return 'Iris Virginica'
    else:
        return 'Error'

# Flask のインスタンスを作成
app = Flask(__name__)

# 入力フォームの設定
class IrisForm(Form):
    SepalLength = FloatField('がくの長さ (0cm ~ 10cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')])

    SepalWidth  = FloatField('がくの幅 (0cm ~ 5cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=5, message='0〜5の数値を入力してください')])

    PetalLength = FloatField('花弁の長さ (0cm ~ 10cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=10, message='0〜10の数値を入力してください')])

    PetalWidth  = FloatField('花弁の幅 (0cm ~ 5cm)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=5, message='0〜5の数値を入力してください')])

    # HTML で表示する submit ボタンの設定
    submit = SubmitField('判定')

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # フォームの設定 IrsiForm 　クラスをインスタンス化
    irisForm = IrisForm(request.form)
    # POST メソッドの定義
    if request.method == 'POST':

        # 条件に当てはまる場合
        if irisForm.validate() == False:
            return render_template('index.html', forms=irisForm)
        # 条件に当てはまらない場合、推論を実行
        else:
            VarSepalLength = float(request.form['SepalLength'])
            VarSepalWidth  = float(request.form['SepalWidth'])
            VarPetalLength = float(request.form['PetalLength'])
            VarPetalWidth  = float(request.form['PetalWidth'])
            # 入力された値を ndarray に変換して推論
            x = np.array([VarSepalLength, VarSepalWidth, VarPetalLength, VarPetalWidth])
            pred = predict(x)
            irisName_ = getName(pred)
            return render_template('result.html', irisName=irisName_)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=irisForm)

# アプリケーションの実行
if __name__ == '__main__':
    app.run()