import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense

# 환경 변수 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 데이터 로드 및 전처리
from main import common_df

common_df["Profit"] = common_df["Profit"].astype(str).str.replace(",", "", regex=False)
common_df["Profit"] = pd.to_numeric(common_df["Profit"], errors="coerce")
common_df["GNI"] = pd.to_numeric(common_df["GNI"], errors="coerce")
common_df["Price"] = pd.to_numeric(common_df["Price"], errors="coerce")
common_df["SMP"] = pd.to_numeric(common_df["SMP"], errors="coerce")
common_df = common_df.dropna()

# 하이퍼파라미터 조정용 변수
variables = ["Profit", "GNI", "Price"]
hyperparameters = {
    "layers": list(range(1, 11)),
    "units": [32, 50, 64],
    "epochs": [30, 50],
    "batch_size": [16, 32],
    "activation": ["relu", "tanh", "sigmoid"],
    "optimizer": ["adam", "sgd", "rmsprop"],
    "loss": ["mse", "mae", "huber_loss"],
}

# 결과 저장
results = []


# LSTM 모델 생성 함수
def create_lstm_model(input_shape, layers, units, activation, optimizer, loss):
    model = Sequential()
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(
            LSTM(
                units,
                activation=activation,
                return_sequences=return_sequences,
                input_shape=input_shape,
            )
        )
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    return model


# 예측 및 평가 함수
def evaluate_model(X_scaled, y_scaled, scaler_y, time_steps, hyperparams):
    layers = hyperparams["layers"]
    units = hyperparams["units"]
    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]
    activation = hyperparams["activation"]
    optimizer = hyperparams["optimizer"]
    loss = hyperparams["loss"]

    # 시계열 데이터 생성
    X_lstm, y_lstm = [], []
    for i in range(len(X_scaled) - time_steps):
        X_lstm.append(X_scaled[i : i + time_steps])
        y_lstm.append(y_scaled[i + time_steps])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    if len(X_lstm) == 0 or len(y_lstm) == 0:
        raise ValueError("Insufficient data to generate time series inputs.")

    # 모델 생성 및 학습
    model = create_lstm_model(
        (X_lstm.shape[1], X_lstm.shape[2]), layers, units, activation, optimizer, loss
    )
    model.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0)

    # 예측
    future_years = [2023, 2024, 2025]
    last_input = X_lstm[-1]
    future_predictions = []

    for _ in future_years:
        next_pred_scaled = model.predict(
            last_input.reshape(1, time_steps, -1), verbose=0
        )
        next_pred = scaler_y.inverse_transform(next_pred_scaled)
        future_predictions.append(next_pred[0][0])
        last_input = np.append(last_input[1:], next_pred_scaled, axis=0)

    return future_predictions


# 하이퍼파라미터 조합 테스트
for var in variables:
    try:
        data = common_df[["Year", var, "SMP"]].sort_values("Year")
        X = data[[var]].values
        y = data["SMP"].values

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        time_steps = 3
        for layers in hyperparameters["layers"]:
            for units in hyperparameters["units"]:
                for epochs in hyperparameters["epochs"]:
                    for batch_size in hyperparameters["batch_size"]:
                        for activation in hyperparameters["activation"]:
                            for optimizer in hyperparameters["optimizer"]:
                                for loss in hyperparameters["loss"]:
                                    predictions = evaluate_model(
                                        X_scaled,
                                        y_scaled,
                                        scaler_y,
                                        time_steps,
                                        {
                                            "layers": layers,
                                            "units": units,
                                            "epochs": epochs,
                                            "batch_size": batch_size,
                                            "activation": activation,
                                            "optimizer": optimizer,
                                            "loss": loss,
                                        },
                                    )
                                    # 정답값 (2023, 2024)
                                    true_values = [167.11, 128.39]
                                    mape = mean_absolute_percentage_error(
                                        true_values[: len(predictions)],
                                        predictions[: len(true_values)],
                                    )
                                    results.append(
                                        {
                                            "Variable": var,
                                            "Layers": layers,
                                            "Units": units,
                                            "Epochs": epochs,
                                            "Batch Size": batch_size,
                                            "Activation": activation,
                                            "Optimizer": optimizer,
                                            "Loss": loss,
                                            "Predictions": predictions,
                                            "MAPE": mape,
                                        }
                                    )
    except Exception as e:
        print(f"Skipping variable {var} due to error: {e}")

# 결과 저장 및 출력
results_df = pd.DataFrame(results)
results_df.to_csv("lstm_results.csv", index=False)
print("Results saved to lstm_results.csv")

# 그래프 시각화
for result in results:
    predictions = result["Predictions"]
    true_values = [167.11, 128.39]
    future_years = [2023, 2024, 2025]

    plt.figure(figsize=(8, 5))
    plt.plot(future_years, predictions, label="Predictions", marker="o")
    plt.plot(
        future_years[: len(true_values)], true_values, label="True Values", marker="o"
    )
    plt.title(f"LSTM Predictions for {result['Variable']}")
    plt.xlabel("Year")
    plt.ylabel("SMP")
    plt.legend()
    plt.grid()
    plt.show()
