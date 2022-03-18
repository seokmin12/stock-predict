from tensorflow.keras.models import load_model
import FinanceDataReader as fdr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

symbol = '011200'


def get_regression(data):
    x_data = list(range(6000, 6000 + 10 * len(data), 10))
    y_data = data

    # X, Y의 평균을 구합니다.
    x_bar = sum(x_data) / len(x_data)
    y_bar = sum(y_data) / len(y_data)

    # 최소제곱법으로 a, b를 구합니다.
    a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(y_data, x_data))])
    a /= sum([(x - x_bar) ** 2 for x in x_data])
    b = y_bar - a * x_bar

    line_x = np.arange(min(x_data), max(x_data), 0.01)
    line_y = a * line_x + b
    if (line_y[-1] - line_y[0]) > 0:
        return 1
    else:
        return 0


def predict():
    predict_start_date = '2022-02-11'
    predict_end_date = '2022-02-17'

    model = load_model(
        '/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/models/stock_model.h5')
    # model = load_model('/Users/seokmin/Desktop/project/stock_first_project/machine_learning_one_symbol/models/011200_model.h5')

    predict_df = fdr.DataReader(symbol, start=predict_start_date, end=predict_end_date)

    data = list(predict_df['Close'])
    plt.plot([1, 2, 3, 4, 5], data)

    predict_data_raw = {'1': data[0],
                        '2': data[1],
                        '3': data[2],
                        '4': data[3],
                        '5': data[4],
                        'regression': get_regression(data)}

    predict_data_set = pd.DataFrame(predict_data_raw, index=[0])

    prediction = model.predict(predict_data_set)
    plt.plot([5, 6, 7, 8, 9], prediction[0], 'r', marker='o')

    return list(prediction[0])


def real_data():
    real_start_date = '2022-02-18'
    real_end_date = '2022-02-24'

    real_df = fdr.DataReader(symbol, start=real_start_date, end=real_end_date)

    data = list(real_df['Close'])

    plt.plot([5, 6, 7, 8, 9], data, 'b', marker='o')

    return data


def error_average():
    prediction = predict()
    real = real_data()

    gab = []

    for i in range(5):
        val = prediction[i] - real[i]
        gab.append(round(val / real[i] * 100, 2))

    print(gab)


if __name__ == "__main__":
    error_average()
    plt.legend(['history', 'prediction', 'real'])
    plt.title(symbol)
    # plt.show()
