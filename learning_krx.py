import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

krx = pd.read_csv('/Users/seokmin/Desktop/project/stock_first_project/machine_learning_one_symbol/csv/krx.csv')
companys = []
for i in krx['code'].tolist():
    if i.isdigit() and len(i) == 6:
        companys.append(i)


def get_data_and_machine_learning():
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

    def get_train_data():
        first_val = []
        sec_val = []
        third_val = []
        fourth_val = []
        fifth_val = []
        regression = []
        result_val = []
        for company in tqdm(companys, desc='Get Train Data'):
            try:
                train_start_day = '2017-01-01'
                train_end_day = '2020-12-31'

                df = fdr.DataReader(company, start=train_start_day, end=train_end_day)

                i = 0

                data_list = []

                price = list(df['Close'])
                for n in range(0, len(price) // 6):
                    val = price[i:i + 6]
                    data_list.append(val)
                    i += 6
                while [] in data_list:
                    data_list.remove([])

                # del data_list[-1]

                for n in data_list:
                    first_val.append(n[0])
                    sec_val.append(n[1])
                    third_val.append(n[2])
                    fourth_val.append(n[3])
                    fifth_val.append(n[4])
                    # 선형 회귀 구하기
                    regression.append(get_regression(n))
                    result_val.append(n[5])

            except IndexError as e:
                continue

        raw = {'1': first_val,
               '2': sec_val,
               '3': third_val,
               '4': fourth_val,
               '5': fifth_val,
               'regression': regression,
               'result': result_val}

        train_set = pd.DataFrame(raw)
        train_set.to_csv(
            '/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/csv/krx_train_data.csv',
            index=False)
        return train_set

    def get_test_data():
        first_val = []
        sec_val = []
        third_val = []
        fourth_val = []
        fifth_val = []
        regression = []
        result_val = []
        for company in tqdm(companys, desc='Get Test Data'):
            try:
                start_day = '2021-01-01'
                end_day = '2022-02-19'

                df = fdr.DataReader(company, start=start_day, end=end_day)

                i = 0

                data_list = []

                price = list(df['Close'])
                for n in range(0, len(price) // 6):
                    val = price[i:i + 6]
                    data_list.append(val)
                    i += 6
                while [] in data_list:
                    data_list.remove([])

                # del data_list[-1]

                for n in data_list:
                    first_val.append(n[0])
                    sec_val.append(n[1])
                    third_val.append(n[2])
                    fourth_val.append(n[3])
                    fifth_val.append(n[4])
                    # 선형 회귀 구하기
                    regression.append(get_regression(n))
                    result_val.append(n[5])
            except IndexError as e:
                continue

        raw = {'1': first_val,
               '2': sec_val,
               '3': third_val,
               '4': fourth_val,
               '5': fifth_val,
               'regression': regression,
               'result': result_val}

        test_set = pd.DataFrame(raw)
        test_set.to_csv(
            '/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/csv/krx_test_data.csv',
            index=False)
        return test_set

    def machine_learning():
        import tensorflow as tf
        from tqdm.keras import TqdmCallback

        train_data = get_train_data()

        x_train = train_data[['1', '2', '3', '4', '5', 'regression']]
        y_train = train_data['result']

        if len(x_train) != len(y_train):
            len_list = [len(x_train), len(y_train)]
            biggest = len_list.index(max(len_list))
            if biggest == 0:
                x_train = x_train.drop(len(train_data) - 1, axis=0)
            if biggest == 1:
                y_train = y_train.drop(len(train_data) - 2, axis=0)

        X = tf.keras.layers.Input(shape=[6])
        H = tf.keras.layers.Dense(16, activation='swish')(X)
        H = tf.keras.layers.Dense(8, activation='swish')(H)
        Y = tf.keras.layers.Dense(1)(H)
        model = tf.keras.models.Model(X, Y)
        model.compile(loss='mse')

        check_point = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/models/stock_model.h5',
            verbose=0,
            save_best_only=False)

        epoch = 100000

        model.fit(x_train, y_train, epochs=epoch, verbose=0, validation_split=0.4,
                  callbacks=[TqdmCallback(verbose=0), check_point])

        return model

    def predict():
        model = machine_learning()
        test_data = get_test_data()

        prediction = model.predict(test_data[['1', '2', '3', '4', '5', 'regression']])
        real = test_data['result']

        gab = []

        for i in range(len(real)):
            real_val = real[i]
            prediction_val = prediction[i]
            gab.append(round(prediction_val - real_val))

        aver = round(sum(gab) / len(gab))

        if aver > 0:
            answer = f'평균적으로 {aver}원 높게 예측됨'
        else:
            answer = f'평균적으로 {aver}원 낮게 예측됨'
        print(answer)
    predict()


def csv_data_and_machine_learning():
    def machine_learning_by_csv():
        import tensorflow as tf
        from tqdm.keras import TqdmCallback

        train_data = pd.read_csv(
            '/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/csv/krx_train_data.csv')

        x_train = train_data[0::2]
        y_train = train_data[1::2]
        y_train = y_train[['1', '2', '3', '4', '5']]

        if len(x_train) != len(y_train):
            len_list = [len(x_train), len(y_train)]
            biggest = len_list.index(max(len_list))
            if biggest == 0:
                x_train = x_train.drop(len(train_data) - 1, axis=0)
            if biggest == 1:
                y_train = y_train.drop(len(train_data) - 2, axis=0)

        X = tf.keras.layers.Input(shape=[6])
        H = tf.keras.layers.Dense(16, activation='swish')(X)
        H = tf.keras.layers.Dense(8, activation='swish')(H)
        Y = tf.keras.layers.Dense(5)(H)
        model = tf.keras.models.Model(X, Y)
        model.compile(loss='mse')

        check_point = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/models/stock_model.h5',
            verbose=0,
            save_best_only=False)

        epoch = 1000

        model.fit(x_train, y_train, epochs=epoch, verbose=0, validation_split=0.4,
                  callbacks=[TqdmCallback(verbose=0), check_point])

        return model

    def predict_by_csv():
        from tqdm.keras import TqdmCallback

        model = machine_learning_by_csv()
        test_data = pd.read_csv(
            '/Users/seokmin/Desktop/project/stock_first_project/machine_learning_every_symbol/csv/krx_test_data.csv')

        prediction = model.predict(test_data[0::2])
        real = test_data[1::2]

        gab = []

        for i in range(len(real)):
            for n in range(5):
                real_val = real.values[i].tolist()[n]
                prediction_val = prediction.tolist()[i][n]
                gab.append(round(prediction_val - real_val))

        aver = round(sum(gab) / len(gab))

        if aver > 0:
            answer = f'평균적으로 {aver}원 높게 예측됨'
        else:
            answer = f'평균적으로 {aver}원 낮게 예측됨'
        print(answer)
    predict_by_csv()


if __name__ == "__main__":
    get_data_and_machine_learning()
    # data가 csv로 있을 때
    # csv_data_and_machine_learning()
