import matplotlib.pyplot as plt

# 예측값과 실제값
predictions = [
    133.53702,
    134.55307,
    134.64604,
    134.39626,
    135.23616,
    136.94331,
    137.01265,
    137.03564,
]
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

# X축 값 (연도)
years = [1, 2, 3, 4, 5, 6, 7, 8]

# 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(years, predictions, label="Predictions")
plt.plot(years, true_values, label="True Values")

# 제목, 레이블 설정
plt.title("Best Configuration SMP Prediction")
plt.xlabel("Year")
plt.ylabel("SMP")
plt.xticks(years)  # X축 값 (연도) 설정
plt.legend()
plt.grid()

# 그래프 저장
plt.savefig("best_lstm_prediction1.png")
plt.close()
