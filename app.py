from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
import pickle
imputer = SimpleImputer()


filename = r'models/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data = [data1,data2,data3,data4,data5,data6,data7,data8]
    data = imputer.fit_transform([data])

    pred = loaded_model.predict(data)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)















