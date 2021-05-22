from flask import Flask, render_template, url_for, flash, redirect, request

# creating an instance of the flask class
app = Flask(__name__)


# HOME
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


# ABOUTUS
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


# PREDICT
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predictalgo', methods=['POST',  'GET'])
def predictalgo():
    comname = request.form['companyname']
    companyname = str(comname)
    import math
    from datetime import date
    import datetime
    import pandas_datareader as crawler
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import re
    today = date.today()
    tomorrow = today + datetime.timedelta(days = 1)
    df = crawler.DataReader(companyname, data_source='yahoo', start='2020-01-01', end=str(today))
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset)*0.8)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    lr = LinearRegression().fit(x_train, y_train)
    lr_prediction = lr.predict(x_test)
    lr_prediction = scaler.inverse_transform(lr_prediction.reshape(-1, 1))
    company_quote = crawler.DataReader(companyname, data_source='yahoo', start='2020-01-01', end=str(today))
    new_df = company_quote.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    pred_price = lr.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price.reshape(-1, 1))
    predictionprice = str(pred_price)
    predictionpricestr = re.sub(r" ?\([^)]+\)", "", predictionprice)

    return render_template('predict.html', pred='Our prediction for {} on {} is {}'.format(companyname, tomorrow, predictionpricestr))


# COMPARE
@app.route('/compare')
def compare():
    return render_template('compare.html')


# PASTPERFORMANCE
@app.route('/pastperformance')
def pastPerformance():
    return render_template('pastPerformance.html')


if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(debug=True)
