import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt


st.title('Stock Price Analysis and Forecasting')

# File uploader to upload the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date']).sort_index()
    df = df[df['Volume'] > 0]
    data = df.loc['1987-10-28':]

    # Define train and validation splits
    df_train = data[data.index < "2019"]
    df_valid = data[data.index >= "2019"]
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Sidebar for navigation
main_option = st.sidebar.selectbox(
    'Select an option',
    ('Visualization', 'Forecasting Models', 'Model Metrics')
)

if main_option == 'Visualization':
    st.subheader('Dataset Visualization')
    st.write(data.head())

    visualization_option = st.selectbox(
        'Select a visualization',
        ('Stock Price over Time', 'Simple Moving Average', 'Exponential Moving Average', 'Relative Strength Index (RSI)', 'Moving Average Convergence Divergence (MACD)', 'Time Series Decomposition', 'Stationarity Test')
    )

    if visualization_option == 'Stock Price over Time':
        st.subheader('Stock Price over Time')
        fig = px.line(data, x="Date", y="Close", title="Closing Price: Range Slider and Selectors")
        fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ))
        st.plotly_chart(fig)

    elif visualization_option == 'Simple Moving Average':
        st.subheader('Simple Moving Average')
        data['SMA_5'] = data['Close'].rolling(5).mean().shift()
        data['SMA_15'] = data['Close'].rolling(15).mean().shift()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data.SMA_5, name='SMA_5'))
        fig.add_trace(go.Scatter(x=data.index, y=data.SMA_15, name='SMA_15'))
        fig.add_trace(go.Scatter(x=data.index, y=data.Close, name='Close', opacity=0.3))
        st.plotly_chart(fig)

    elif visualization_option == 'Exponential Moving Average':
        st.subheader('Exponential Moving Average')
        data['EMA_5'] = data['Close'].ewm(5).mean().shift()
        data['EMA_15'] = data['Close'].ewm(15).mean().shift()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data.EMA_5, name='EMA_5'))
        fig.add_trace(go.Scatter(x=data.index, y=data.EMA_15, name='EMA_15'))
        fig.add_trace(go.Scatter(x=data.index, y=data.Close, name='Close', opacity=0.3))
        st.plotly_chart(fig)

    elif visualization_option == 'Relative Strength Index (RSI)':
        st.subheader('Relative Strength Index (RSI)')
        def RSI(df, n=14):
            close = df['Close']
            delta = close.diff()
            delta = delta[1:]
            pricesUp = delta.copy()
            pricesDown = delta.copy()
            pricesUp[pricesUp < 0] = 0
            pricesDown[pricesDown > 0] = 0
            rollUp = pricesUp.rolling(n).mean()
            rollDown = pricesDown.abs().rolling(n).mean()
            rs = rollUp / rollDown
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

        data['RSI'] = RSI(df).fillna(0)
        fig = go.Figure(go.Scatter(x=data.index, y=data.RSI, name='RSI'))
        st.plotly_chart(fig)

    elif visualization_option == 'Moving Average Convergence Divergence (MACD)':
        st.subheader('Moving Average Convergence Divergence (MACD)')
        data['EMA_12'] = pd.Series(data['Close'].ewm(span=12).mean())
        data['EMA_26'] = pd.Series(data['Close'].ewm(span=26).mean())
        data['MACD'] = pd.Series(data['EMA_12'] - data['EMA_26'])
        data['MACD_signal'] = pd.Series(data.MACD.ewm(span=9, min_periods=9).mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data.MACD, name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data.MACD_signal, name='MACD_signal'))
        st.plotly_chart(fig)

    elif visualization_option == 'Time Series Decomposition':
        st.subheader('Time Series Decomposition')
        series = data.Close
        result = seasonal_decompose(series, model='additive', period=365)
        fig = result.plot()
        st.pyplot(fig)

    elif visualization_option == 'Stationarity Test':
        st.subheader('Stationarity Test')
        def test_stationarity(timeseries):
            rolmean = timeseries.rolling(12).mean()
            rolstd = timeseries.rolling(12).std()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, mode='lines', name='Original'))
            fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean, mode='lines', name='Rolling Mean'))
            fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd, mode='lines', name='Rolling Std'))
            fig.update_layout(title='Rolling Mean and Standard Deviation', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)

            st.write("Results of Dickey Fuller Test")
            adft = adfuller(timeseries, autolag='AIC')
            output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
            st.write(output)

        test_stationarity(data['Close'])

elif main_option == 'Forecasting Models':
    st.subheader('Forecasting Models')

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    forecasting_model = st.selectbox(
        'Select a forecasting model',
        ('Select a model', 'ARIMA Forecast', 'SARIMAX Forecast', 'ETS Forecast', 'Prophet Forecast', 'SVR Forecast', 'Hybrid ARIMA + ANN Forecast', 'LSTM Forecast', 'Simple ANN Forecast')
    )

    if forecasting_model != 'Select a model':
        if st.session_state.selected_model != forecasting_model:
            st.session_state.selected_model = forecasting_model
            st.experimental_rerun()

    if st.session_state.selected_model == 'ARIMA Forecast':
        st.subheader('ARIMA Model')
        train = df_train['Close'].values
        test = df_valid['Close'].values

        # Rolling ARIMA
        history = [x for x in train]
        predictions = list()
        for t in range(len(df_valid)):
            model = ARIMA(history, order=(3, 1, 3))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        rolling_mse = mean_squared_error(test, predictions)
        st.write('Test MSE (Rolling ARIMA): %.3f' % rolling_mse)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid.Close, name='Close'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=predictions, name='Forecast_Rolling_ARIMA'))
        st.plotly_chart(fig)


    elif st.session_state.selected_model == 'SARIMAX Forecast':
        st.subheader('SARIMAX Model')
        df_train_weekly = df_train['Close'].resample('W').mean()
        df_valid_weekly = df_valid['Close'].resample('W').mean()

        model = SARIMAX(df_train_weekly, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52), enforce_stationarity=True, enforce_invertibility=True)
        model_fit = model.fit(disp=False)

        n_periods = len(df_valid_weekly)
        forecast = model_fit.get_forecast(steps=n_periods)
        forecast_index = df_valid_weekly.index
        forecast_series = forecast.predicted_mean
        conf_int = forecast.conf_int()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train_weekly.index, y=df_train_weekly, mode='lines', name='Training Data (Weekly Avg)'))
        fig.add_trace(go.Scatter(x=df_valid_weekly.index, y=df_valid_weekly, mode='lines', name='Actual Validation Data'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_series, mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], mode='lines', line=dict(color="#444"), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], mode='lines', line=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty', showlegend=False, name='Confidence Interval'))
        st.plotly_chart(fig)

    elif st.session_state.selected_model == 'ETS Forecast':
        st.subheader('ETS Model')
        ets_model = ExponentialSmoothing(df_train['Close'], trend='add', seasonal='mul', seasonal_periods=365).fit()
        ets_forecast = ets_model.forecast(steps=len(df_valid))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index[-365:], y=df_train['Close'][-365:], mode='lines', name='Train (Last Year)'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=ets_forecast, mode='lines', name='ETS Forecast'))
        st.plotly_chart(fig)

    elif st.session_state.selected_model == 'Prophet Forecast':
        st.subheader('Prophet Model')
        df_prophet = df_train['Close'].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        prophet_model.fit(df_prophet)

        future = prophet_model.make_future_dataframe(periods=len(df_valid), freq='D')
        prophet_forecast = prophet_model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Close'], mode='lines', name='Historical Close'))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill=None, mode='lines', line=dict(color='lightgrey'), showlegend=False))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='lightgrey'), showlegend=False))
        st.plotly_chart(fig)

    elif st.session_state.selected_model == 'SVR Forecast':
        st.subheader('SVR Model')
        train_shifted = df_train['Close'].shift(1).fillna(method='bfill')
        X_train = train_shifted.values.reshape(-1, 1)
        y_train = df_train['Close'].values
        X_test = df_valid['Close'].shift(1).fillna(method='bfill').values.reshape(-1, 1)

        svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel='rbf'))
        svr_model.fit(X_train, y_train)
        svr_predictions = svr_model.predict(X_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Close'], mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=svr_predictions, mode='lines', name='SVR Forecast'))
        st.plotly_chart(fig)

        svr_mse = mean_squared_error(df_valid['Close'], svr_predictions)
        st.write('Test MSE (SVR): %.3f' % svr_mse)

    elif st.session_state.selected_model == 'Hybrid ARIMA + ANN Forecast':
        st.subheader('Hybrid ARIMA + ANN Model')

        # Differencing function
        def difference(dataset, interval=1):
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return np.array(diff)

        # Inverse differencing function
        def inverse_difference(history, yhat, interval=1):
            return yhat + history[-interval]

        train = df_train['Close'].values
        test = df_valid['Close'].values

        # Perform differencing
        days_in_year = 365
        differenced = difference(train, days_in_year)

        # Fit ARIMA model
        model = ARIMA(differenced, order=(10, 1, 4))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

        # Invert the differenced forecast to the original scale
        history = [x for x in train]
        forecast = [inverse_difference(history, yhat, days_in_year) for yhat in forecast]

        # Calculate residuals
        residuals = [test[i] - forecast[i] for i in range(len(test))]

        # Prepare data for ANN
        X_train = np.arange(len(residuals)).reshape(-1, 1)
        y_train = np.array(residuals)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit ANN model
        ann_model = Sequential()
        ann_model.add(Dense(100, activation='relu', input_shape=(1,)))
        ann_model.add(Dense(100, activation='relu'))
        ann_model.add(Dense(1))
        ann_model.compile(optimizer='adam', loss='mean_squared_error')
        ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

        # Predict corrections
        corrections = ann_model.predict(X_train_scaled)

        # Final corrected forecast
        final_forecast = [forecast[i] + corrections[i] for i in range(len(forecast))]

        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=final_forecast, name='Corrected Forecast (ARIMA + ANN)'))
        st.plotly_chart(fig)

        hybrid_mse = mean_squared_error(df_valid['Close'], final_forecast)
        st.write('Test MSE (Hybrid ARIMA + ANN): %.3f' % hybrid_mse)

    elif st.session_state.selected_model == 'LSTM Forecast':
        st.subheader('LSTM Model')
        train = df_train['Close'].values
        test = df_valid['Close'].values

        # Data preprocessing
        training_values = np.reshape(train, (len(train), 1))
        scaler = MinMaxScaler()
        training_values = scaler.fit_transform(training_values)

        x_train = training_values[0:len(training_values) - 1]
        y_train = training_values[1:len(training_values)]
        x_train = np.reshape(x_train, (len(x_train), 1, 1))

        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, epochs=25, batch_size=8)

        # Prepare test data
        test_values = np.reshape(test, (len(test), 1))
        test_values = scaler.transform(test_values)
        test_values = np.reshape(test_values, (len(test_values), 1, 1))

        # Make predictions
        predicted_price = model.predict(test_values)
        predicted_price = scaler.inverse_transform(predicted_price)
        predicted_price = np.squeeze(predicted_price)

        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df_valid.index, y=predicted_price, name='Forecast_LSTM'))
        st.plotly_chart(fig)

        # Calculate MSE
        mse_lstm = mean_squared_error(test, predicted_price)
        st.write('Test MSE (LSTM): %.3f' % mse_lstm)

    elif st.session_state.selected_model == 'Simple ANN Forecast':
        st.subheader('Simple ANN Model')
        train = df_train['Close'].values
        test = df_valid['Close'].values

        # Prepare data for ANN
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(n_steps, len(data)):
                X.append(data[i-n_steps:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        n_steps = 10  # Number of steps to look back
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1))
        test_scaled = scaler.transform(test.reshape(-1, 1))

        x_train, y_train = create_sequences(train_scaled, n_steps)
        x_test, y_test = create_sequences(test_scaled, n_steps)

        # Model creation
        def create_model(input_shape):
            model = Sequential()
            model.add(Dense(100, activation='relu', input_shape=(input_shape,)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(100, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
            return model

        model = create_model(n_steps)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        # Train the model
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                            epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

        # Predict and plot
        predictions = model.predict(x_test)
        fig=plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Values')
        plt.legend()
        st.write(fig)


        # Calculate MSE
        mse_ann = mean_squared_error(y_test, predictions)
        st.write('Test MSE (Simple ANN): %.3f' % mse_ann)

elif main_option == 'Model Metrics':
    st.subheader('Model Metrics')

    metrics = {}

    # ARIMA Model
    train = df_train['Close'].values
    test = df_valid['Close'].values

    # Rolling ARIMA
    history = [x for x in train]
    predictions = list()
    for t in range(len(df_valid)):
        model = ARIMA(history, order=(3, 1, 3))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    rolling_mse = mean_squared_error(test, predictions)
    metrics['ARIMA (Rolling)'] = rolling_mse


    # SARIMAX Model
    df_train_weekly = df_train['Close'].resample('W').mean()
    df_valid_weekly = df_valid['Close'].resample('W').mean()
    model = SARIMAX(df_train_weekly, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52), enforce_stationarity=True, enforce_invertibility=True)
    model_fit = model.fit(disp=False)
    n_periods = len(df_valid_weekly)
    forecast = model_fit.get_forecast(steps=n_periods)
    forecast_series = forecast.predicted_mean
    sarimax_mse = mean_squared_error(df_valid_weekly, forecast_series)
    metrics['SARIMAX'] = sarimax_mse

    # ETS Model
    ets_model = ExponentialSmoothing(df_train['Close'], trend='add', seasonal='mul', seasonal_periods=365).fit()
    ets_forecast = ets_model.forecast(steps=len(df_valid))
    ets_mse = mean_squared_error(df_valid['Close'], ets_forecast)
    metrics['ETS'] = ets_mse

    # Prophet Model
    df_prophet = df_train['Close'].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=len(df_valid), freq='D')
    prophet_forecast = prophet_model.predict(future)
    prophet_mse = mean_squared_error(df_valid['Close'], prophet_forecast['yhat'][-len(df_valid):])
    metrics['Prophet'] = prophet_mse

    # SVR Model
    train_shifted = df_train['Close'].shift(1).fillna(method='bfill')
    X_train = train_shifted.values.reshape(-1, 1)
    y_train = df_train['Close'].values
    X_test = df_valid['Close'].shift(1).fillna(method='bfill').values.reshape(-1, 1)
    svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel='rbf'))
    svr_model.fit(X_train, y_train)
    svr_predictions = svr_model.predict(X_test)
    svr_mse = mean_squared_error(df_valid['Close'], svr_predictions)
    metrics['SVR'] = svr_mse

    # Hybrid ARIMA + ANN Model
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)

    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    days_in_year = 365
    differenced = difference(train, days_in_year)
    model = ARIMA(differenced, order=(10, 1, 4))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    history = [x for x in train]
    forecast = [inverse_difference(history, yhat, days_in_year) for yhat in forecast]
    residuals = [test[i] - forecast[i] for i in range(len(test))]
    X_train = np.arange(len(residuals)).reshape(-1, 1)
    y_train = np.array(residuals)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ann_model = Sequential()
    ann_model.add(Dense(100, activation='relu', input_shape=(1,)))
    ann_model.add(Dense(100, activation='relu'))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mean_squared_error')
    ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
    corrections = ann_model.predict(X_train_scaled)
    final_forecast = [forecast[i] + corrections[i] for i in range(len(forecast))]
    hybrid_mse = mean_squared_error(df_valid['Close'], final_forecast)
    metrics['Hybrid ARIMA + ANN'] = hybrid_mse

    # LSTM Model
    training_values = np.reshape(train, (len(train), 1))
    scaler = MinMaxScaler()
    training_values = scaler.fit_transform(training_values)
    x_train = training_values[0:len(training_values) - 1]
    y_train = training_values[1:len(training_values)]
    x_train = np.reshape(x_train, (len(x_train), 1, 1))
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=8)
    test_values = np.reshape(test, (len(test), 1))
    test_values = scaler.transform(test_values)
    test_values = np.reshape(test_values, (len(test_values), 1, 1))
    predicted_price = model.predict(test_values)
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_price = np.squeeze(predicted_price)
    mse_lstm = mean_squared_error(test, predicted_price)
    metrics['LSTM'] = mse_lstm

    # Simple ANN Model
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i-n_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    n_steps = 10  # Number of steps to look back
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    x_train, y_train = create_sequences(train_scaled, n_steps)
    x_test, y_test = create_sequences(test_scaled, n_steps)

    # Model creation
    def create_model(input_shape):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(input_shape,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        return model

    model = create_model(n_steps)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Predict and plot
    predictions = model.predict(x_test)
    mse_ann = mean_squared_error(y_test, predictions)
    metrics['Simple ANN'] = mse_ann

    # Display Metrics
    st.write("Model Metrics:")
    for model_name, mse in metrics.items():
        st.write(f"{model_name}: Test MSE: {mse:.3f}")
