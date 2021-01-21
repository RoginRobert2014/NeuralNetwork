import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# import training data
df_train = pd.read_csv("train.csv")  # import data to pandas dataframe
pd.to_datetime(df_train.Date)  # change data type of date column
del df_train['Date']  # remove Date column
del df_train['Customers']  # remove Customers column
stores = df_train['Store'].unique().tolist()  # creates a list of all stores

# import prediction data
df_pred = pd.read_csv("test.csv")  # import prediction data to pandas dataframe
del df_pred['Date']  # remove Date column
pred_stores = df_pred['Store'].unique().tolist()  # creates list of all stores to predict

for store in pred_stores:  # iterate over dat for each store
    store_LSTM_data = df_train[(df_train['Store'] == store)]  # data to build LSTM on
    y_col = 'Sales'  # specifies target column

    # split dta into train/test
    test_size = int(len(store_LSTM_data) * 0.2)  # test set is 20% of data
    train = store_LSTM_data.iloc[:-test_size, :].copy()  # gets first 80% of data
    test = store_LSTM_data.iloc[-test_size:, :].copy()  # gets last 20% of data

    x_train = train.drop(['Store', y_col], axis=1).copy()  # removes store number and sales column from data
    y_train = train[[y_col]].copy()  # obtains target data

    # normalize data
    x_scaler = MinMaxScaler(feature_range=(0, 1))  # sets range for normalising
    x_scaler.fit(x_train)  # shape manipulation
    scaled_x_train = x_scaler.transform(x_train)  # creates scaled data for feature variables

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_train)  # shape manipulation
    scaled_y_train = y_scaler.transform(y_train)  # creates scaled data for target values
    scaled_y_train = scaled_y_train.reshape(-1)  # need sales column in format (n,) not (n, 1)

    n_input = 20  # how many timesteps to look in the past to generate forecast
    n_features = x_train.shape[1]  # number of predictor variables
    b_size = 32  # Number of timeseries samples in each batch
    generator = TimeseriesGenerator(scaled_x_train, scaled_y_train, length=n_input, batch_size=b_size)  # creates data structure for DNN

    # initialise model - need to test and develop
    model = Sequential()  # created model pattern
    model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))  # adds LSTM layer
    model.add(Dense(100, activation='relu'))  # MLP style layer with 100 neurons
    model.add(Dense(50, activation='relu'))  # MLP style layer with 50 neurons
    model.add(Dense(25, activation='relu'))  # MLP style layer with 25 neurons
    model.add(Dense(20, activation='relu'))  # MLP style layer with 20 neurons
    model.add(Dense(10, activation='relu'))  # MLP style layer with 10 neurons
    model.add(Dense(1))  # adds output layer
    model.compile(optimizer='adam', loss='mse')  # compiles model
    model.summary()  # gives summary of model describing shapes of outputs

    model.fit(generator, epochs=500)  # carries out training of model

    # checking accuracy
    x_test = test.drop(['Store', y_col], axis=1).copy()  # removes store number and sales column from data
    scaled_x_test = x_scaler.transform(x_test)  # scales data
    test_generator = TimeseriesGenerator(scaled_x_test, np.zeros(len(x_test)), length=n_input, batch_size=b_size)  # creates time series data for DNN

    y_pred_scaled = model.predict(test_generator)  # calls predict method
    y_pred = y_scaler.inverse_transform(y_pred_scaled)  # rescales predictions
    results = pd.DataFrame({'y_true': test[y_col].values[n_input:], 'y_pred': y_pred.ravel()})  # creates dataframe from results by concatenating y_pred results

    # predict
    pred_df = df_pred[(df_pred.Store == store)]  # data to build prediction on
    dfs = [x_test[n_input: -1], pred_df]  # creates list of preliminary data
    store_LSTM_pred_data = pd.concat(dfs)  # attaches preliminary data to data to be predicted
    features = store_LSTM_pred_data[['DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']].copy()  # creates copy of data to be predicted from columns
    scaled_features = x_scaler.transform(features)  # normalise data to be predicted

    test_generator = TimeseriesGenerator(scaled_features, np.zeros(len(features)), length=n_input, batch_size=b_size)  # creates data structure for DNN

    scaled_pred = model.predict(test_generator)  # run the data through the neural network
    pred = y_scaler.inverse_transform(scaled_pred)  # denormalise data
    results_pred = pd.DataFrame({'Id': pred_df['Id'], 'y_predicted': pred.ravel()[-48:]})  # take final 48 entries
    results_pred.to_csv('predicted sales.csv', mode='a', header=False)  # save results with corresponding Id number in csv file
