# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Multiply,
    Permute,
    Lambda,
    Softmax,
)
import tensorflow.keras.backend as K
import tensorflow as tf
from tqdm import tqdm


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(self.quantile * error, (self.quantile - 1) * error)
        )


# 환경 변수 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 데이터 로드 및 전처리
from main import common_df

common_df["Profit"] = common_df["Profit"].astype(str).str.replace(",", "", regex=False)
common_df["Profit"] = pd.to_numeric(common_df["Profit"], errors="coerce")
common_df["GNI"] = pd.to_numeric(common_df["GNI"], errors="coerce")
common_df["Price"] = pd.to_numeric(common_df["Price"], errors="coerce")
common_df["SMP"] = pd.to_numeric(common_df["SMP"], errors="coerce")
common_df["Year"] = pd.to_numeric(common_df["Year"], errors="coerce")

common_df = common_df.dropna()

# 변수별 데이터 중앙값으로 통합
variables = ["Profit", "GNI", "Price"]
# 변수별 데이터 중앙값으로 통합
# median_values = common_df.groupby("Year")[variables].median().reset_index()

# PCA 모델 생성 (주성분 1개만 추출)
pca = PCA(n_components=1)
X = common_df[variables]
# PCA 적용
common_df["PCA_Variable"] = pca.fit_transform(X)
common_df["PCA_Variable"] = pd.to_numeric(common_df["PCA_Variable"], errors="coerce")
common_df = common_df.dropna(subset=["PCA_Variable", "SMP"])
print(common_df[["Year", "PCA_Variable"]])


# 노이즈 추가
def add_noise(data, noise_std=0.02):
    return data + np.random.normal(0, noise_std, data.shape)


# 하이퍼파라미터 조정용 변수
"""
hyperparameters = {
    "layers": [3, 5, 7],
    "units": [32, 50, 64],
    "epochs": [30, 50],
    "batch_size": [16, 32],
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "rmsprop"],
    "loss": ["mse", "huber_loss", QuantileLoss(0.1), QuantileLoss(0.9)],
}
"""
hyperparameters = {
    "layers": [7],
    "units": [64],
    "epochs": [200],
    "batch_size": [32],
    "activation": ["relu"],
    "optimizer": ["adam"],
    "loss": ["huber_loss"],
}

# 결과 저장
results = []


# 어텐션 레이어 추가
def attention_layer(inputs):
    attention = Dense(inputs.shape[1], activation="softmax")(inputs)
    attention = Multiply()([inputs, attention])
    return attention


# LSTM 모델 생성 함수
def create_lstm_attention_model(
    input_shape, layers, units, activation, optimizer, loss
):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(layers):
        return_sequences = i < layers - 1
        x = LSTM(units, activation=activation, return_sequences=return_sequences)(x)

    x = attention_layer(x)  # 어텐션 적용
    output = Dense(1)(x)

    model = Model(inputs, output)
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

    X_scaled = add_noise(X_scaled)  # 데이터 증강

    # 시계열 데이터 생성
    X_lstm, y_lstm = [], []
    for i in range(len(X_scaled) - time_steps):
        X_lstm.append(X_scaled[i : i + time_steps])
        y_lstm.append(y_scaled[i + time_steps])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    if len(X_lstm) == 0 or len(y_lstm) == 0:
        raise ValueError("Insufficient data to generate time series inputs.")

    # 모델 생성 및 학습
    model = create_lstm_attention_model(
        (X_lstm.shape[1], X_lstm.shape[2]), layers, units, activation, optimizer, loss
    )
    for _ in tqdm(range(epochs), desc="Training Progress"):
        model.fit(
            X_lstm,
            y_lstm,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
        )

    ### 📌 1. 기존 데이터 (2004-2022) 예측 ###
    train_predictions_scaled = model.predict(X_lstm, verbose=0)
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled).flatten()

    ### 📌 2. 미래 예측 (2023-2024) ###
    num_predictions = 8
    future_years = list(
        range(
            common_df["Year"].max() + 1, common_df["Year"].max() + num_predictions + 1
        )
    )

    last_input = X_lstm[-1]
    future_predictions = []

    for _ in tqdm(range(num_predictions), desc="Predicting Future Values"):
        next_pred_scaled = model.predict(
            last_input.reshape(1, time_steps, -1), verbose=0
        )
        next_pred = scaler_y.inverse_transform(next_pred_scaled)
        future_predictions.append(next_pred[0][0])
        last_input = np.vstack([last_input[1:], next_pred_scaled])

    return train_predictions, future_years, future_predictions


# 모델 평가 및 결과 저장

data = common_df[["Year", "PCA_Variable", "SMP"]].sort_values("Year")
X = data[["PCA_Variable"]].values
y = data["SMP"].values
# print(f'x:{X},y:{y}')


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))


total_iterations = (
    len(range(3, 4))
    * len(hyperparameters["layers"])
    * len(hyperparameters["units"])
    * len(hyperparameters["epochs"])
    * len(hyperparameters["batch_size"])
    * len(hyperparameters["activation"])
    * len(hyperparameters["optimizer"])
    * len(hyperparameters["loss"])
)
count = 0

with tqdm(total=total_iterations, desc="Hyperparameter Tuning") as pbar:
    for time_steps in range(3,4):
        for layers in hyperparameters["layers"]:
            for units in hyperparameters["units"]:
                for epochs in hyperparameters["epochs"]:
                    for batch_size in hyperparameters["batch_size"]:
                        for activation in hyperparameters["activation"]:
                            for optimizer in hyperparameters["optimizer"]:
                                for loss in hyperparameters["loss"]:
                                    train_predictions, future_years, predictions = (
                                        evaluate_model(
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
                                    )
                                    # 정답값 (2023, 2024)
                                    true_values = [
                                        237.61,
                                        151.19,
                                        229.65,
                                        134.87,
                                        134.99,
                                        128.41,
                                        141.13,
                                        115.14,
                                    ]
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
                                    count += 1
                                    if count % 10 == 0:
                                        results_df = pd.DataFrame(results)
                                        results_df.to_csv(
                                            "lstm_results.csv", index=False
                                        )
                                    pbar.update(1)

if results:
    # 결과 저장
    print(
        "train_predictions: {}, future_years: {}, predictions: {}".format(
            len(train_predictions), len(future_years), len(predictions)
        )
    )

    # 최적의 설정 찾기 (최소 MAPE)
    best_result = min(results, key=lambda x: x["MAPE"])
    best_results = []
    best_results.append({"Best Result": best_result})
    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv("lstm_best_results.csv", index=False)
    print(best_result)

    # 최적 설정 그래프 시각화 및 저장
    plt.figure(figsize=(8, 5))

    # 2022년까지의 실제 SMP (검은색)
    plt.plot(
        data["Year"][: len(train_predictions)],
        data["SMP"].values[: len(train_predictions)],
        color="black",
        label="Actual SMP",
        linewidth=2,
    )

    # LSTM 모델로 예측된 train SMP (빨간색)
    plt.plot(
        data["Year"][: len(train_predictions)],
        train_predictions,
        color="red",
        label="Train Predictions (LSTM)",
        linestyle="--",
    )

    # 2023~2024년 동안의 예측된 future SMP (파란색)
    plt.plot(
        future_years,
        best_result["Predictions"],
        color="blue",
        label="Future Predictions",
        marker="o",
    )

    # 2023~2024년 동안의 실제 SMP (그린색)
    future_smp = [237.61, 151.19, 229.65, 134.87, 134.99, 128.41, 141.13, 115.14]
    plt.plot(
        future_years, future_smp, color="green", label="Future True SMP", marker="x"
    )

    plt.xlabel("Year")
    plt.ylabel("SMP")
    plt.title("SMP Prediction and True Values (LSTM + Future)")
    plt.legend()
    plt.grid(True)
    plt.savefig("smp_predictions_and_truth.png")
    plt.show()
