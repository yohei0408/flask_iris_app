from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import joblib

iris = load_iris()

x = iris.data
t = iris.target

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

model = svm.LinearSVC()

model.fit(x_train, t_train)
pred = model.predict(x_test)

print(classification_report(t_test, pred))

joblib.dump(model, "src/iris.pkl", compress=True)