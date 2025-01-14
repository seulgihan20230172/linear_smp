import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense
from main import common_df


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 데이터 전처리
common_df["Profit"] = common_df["Profit"].astype(str).str.replace(",", "", regex=False)
common_df["Profit"] = pd.to_numeric(common_df["Profit"], errors="coerce")
common_df["GNI"] = pd.to_numeric(common_df["GNI"], errors="coerce")
common_df["Price"] = pd.to_numeric(common_df["Price"], errors="coerce")
common_df["SMP"] = pd.to_numeric(common_df["SMP"], errors="coerce")
common_df = common_df.dropna()

# 변수 목록
variables = ["Profit", "GNI", "Price"]
final_smp_predictions = {}

# 실제값
actual_values = {2023: 167.11, 2024: 128.39}


# LSTM 모델 생성 함수
def create_lstm_model(input_shape, neurons=50, layers=2):
    model = Sequential()
    # 첫 번째 LSTM 레이어
    model.add(
        LSTM(neurons, activation="relu", return_sequences=True, input_shape=input_shape)
    )

    # 추가 LSTM 레이어
    for _ in range(layers - 1):
        model.add(LSTM(neurons, activation="relu"))

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


# 하이퍼파라미터 조합 설정
param_combinations = [
    {"neurons": 50, "layers": 2, "batch_size": 16, "epochs": 50},
    {"neurons": 100, "layers": 2, "batch_size": 32, "epochs": 100},
    {"neurons": 50, "layers": 3, "batch_size": 16, "epochs": 100},
    {"neurons": 64, "layers": 2, "batch_size": 64, "epochs": 50},
]


# 정확도를 기록할 변수
best_accuracy = float("-inf")
best_configuration = {}

# 변수별로 분석
for var in variables:
    try:
        # 독립 변수와 종속 변수 준비
        data = common_df[["Year", var, "SMP"]].sort_values("Year")
        X = data[[var]].values  # 독립 변수
        y = data["SMP"].values  # 종속 변수

        # 데이터 스케일링
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # 시계열 데이터 생성 (LSTM 입력 형식: [samples, timesteps, features])
        time_steps = 3
        X_lstm, y_lstm = [], []
        for i in range(len(X_scaled) - time_steps):
            X_lstm.append(X_scaled[i : i + time_steps])
            y_lstm.append(y_scaled[i + time_steps])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        # LSTM 모델 학습
        for params in param_combinations:
            model = create_lstm_model(
                input_shape=(X_lstm.shape[1], X_lstm.shape[2]),
                neurons=params["neurons"],
                layers=params["layers"],
            )

            # 모델 학습
            model.fit(
                X_lstm,
                y_lstm,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
            )

            # 재귀 예측을 통해 2023년과 2024년 예측
            future_years = [2023, 2024]
            last_input = X_lstm[-1]
            future_predictions = {}

        for year in future_years:
            next_pred_scaled = model.predict(last_input.reshape(1, time_steps, -1))
            next_pred = scaler_y.inverse_transform(next_pred_scaled)
            future_predictions[year] = next_pred[0][0]
            # 새로운 입력 데이터 준비
            next_input = np.append(last_input[1:], next_pred_scaled, axis=0)
            last_input = next_input

        # 예측 정확도 계산
        accuracy_2023 = (
            100
            - abs(
                (future_predictions[2023] - actual_values[2023]) / actual_values[2023]
            )
            * 100
        )
        accuracy_2024 = (
            100
            - abs(
                (future_predictions[2024] - actual_values[2024]) / actual_values[2024]
            )
            * 100
        )

        avg_accuracy = (accuracy_2023 + accuracy_2024) / 2
        print(
            f"Variable: {var}, 2023 Accuracy: {accuracy_2023:.2f}%, 2024 Accuracy: {accuracy_2024:.2f}%"
        )

        # 최고 정확도 업데이트
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_configuration = {
                "variable": var,
                "accuracy_2023": accuracy_2023,
                "accuracy_2024": accuracy_2024,
                "avg_accuracy": avg_accuracy,
            }

        # 중앙값으로 해당 변수의 SMP 설정
        predicted_smp = np.median(list(future_predictions.values()))
        final_smp_predictions[var] = predicted_smp

        # 시각화
        plt.plot(
            data["Year"].iloc[time_steps:],
            scaler_y.inverse_transform(model.predict(X_lstm)),
            label="Predicted SMP",
            color="green",
        )
        plt.title(f"LSTM SMP Prediction for {var}")
        plt.xlabel("Year")
        plt.ylabel("SMP")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Skipping variable {var} due to error: {e}")

# 최종 결과 병합
future_years = [2023, 2024, 2025]
final_df = pd.DataFrame(
    {
        "Year": future_years,
        "Predicted SMP": [final_smp_predictions.get(var, np.nan) for var in variables],
    }
)
print(final_df)

# 최고 정확도를 기록한 변수 출력
print("\nBest Configuration based on accuracy:")
print(best_configuration)
