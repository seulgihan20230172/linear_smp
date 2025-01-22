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

# 변수별 데이터 중앙값으로 통합
variables = ["Profit", "GNI", "Price"]
# 변수별 데이터 중앙값으로 통합
median_values = common_df.groupby("Year")[variables].median().reset_index()
common_df = common_df.merge(median_values, on="Year", suffixes=("", "_Median"))
common_df["Median_Variable"] = common_df[[f"{var}_Median" for var in variables]].median(
    axis=1
)
# Median_Variable 열을 숫자형으로 변환
common_df["Median_Variable"] = pd.to_numeric(
    common_df["Median_Variable"], errors="coerce"
)

# NaN 값 제거
common_df = common_df.dropna(subset=["Median_Variable", "SMP"])

# Median_Variable 확인
print(common_df[["Year", "Median_Variable"]].head())


# 하이퍼파라미터 조정용 변수
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
    future_years = list(range(common_df["Year"].min(), 2026))
    last_input = X_lstm[-1]
    future_predictions = []

    for _ in future_years[len(X_lstm) :]:
        next_pred_scaled = model.predict(
            last_input.reshape(1, time_steps, -1), verbose=0
        )
        next_pred = scaler_y.inverse_transform(next_pred_scaled)
        future_predictions.append(next_pred[0][0])
        last_input = np.append(last_input[1:], next_pred_scaled, axis=0)

    return future_years, future_predictions


# 모델 평가 및 결과 저장
try:
    data = common_df[["Year", "Median_Variable", "SMP"]].sort_values("Year")
    X = data[["Median_Variable"]].values
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
                                future_years, predictions = evaluate_model(
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
    print(f"Error occurred: {e}")

if results:
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("lstm_results.csv", index=False)

    # 최적의 설정 찾기 (최소 MAPE)
    best_result = min(results, key=lambda x: x["MAPE"])

    # 최적 설정 그래프 시각화 및 저장
    plt.figure(figsize=(8, 5))
    plt.plot(best_result["Predictions"], label="Predictions", marker="o")
    plt.plot([167.11, 128.39], label="True Values", marker="o")
    plt.title(
        f"Best Configuration: Layers={best_result['Layers']}, Units={best_result['Units']}, Epochs={best_result['Epochs']}"
    )
    plt.xlabel("Year")
    plt.ylabel("SMP")
    plt.legend()
    plt.grid()
    plt.savefig("best_lstm_prediction.png")
    plt.close()

    print("Results saved to lstm_results.csv and best_lstm_prediction.png")

else:
    print("No results were generated.")
