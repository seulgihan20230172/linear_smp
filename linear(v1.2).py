import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
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


# LSTM 모델 생성 함수
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(
        LSTM(50, activation="relu", return_sequences=True, input_shape=input_shape)
    )
    model.add(LSTM(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


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
        model = create_lstm_model(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
        model.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0)

        # 재귀 예측을 통해 2025년까지 예측
        future_years = [2023, 2024, 2025]
        last_input = X_lstm[-1]
        future_predictions = []

        for _ in future_years:
            next_pred_scaled = model.predict(last_input.reshape(1, time_steps, -1))
            next_pred = scaler_y.inverse_transform(next_pred_scaled)
            future_predictions.append(next_pred[0][0])
            # 새로운 입력 데이터 준비
            next_input = np.append(last_input[1:], next_pred_scaled, axis=0)
            last_input = next_input

        # 중앙값으로 해당 변수의 SMP 설정
        predicted_smp = np.median(future_predictions)
        final_smp_predictions[var] = predicted_smp
        """
        # 시각화
        plt.figure(figsize=(16, 9))
        plt.plot(
            data["Year"].iloc[time_steps:],
            y[time_steps:],
            label="Actual SMP",
            color="blue",
        )
        """
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
