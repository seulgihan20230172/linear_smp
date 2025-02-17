import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.getcwd())  # 현재 디렉터리 출력

# 상대 경로로 파일 불러오기 (현재 디렉토리에 있는 경우)
file_path = "linear_smp/lstm_results.csv"

# CSV 파일 로드
df = pd.read_csv(file_path)

# MAPE 값이 가장 작은 행 선택
best_result = df.loc[df["MAPE"].idxmin()]

# Predictions 값 가져오기 (문자열 형태일 가능성이 있어 변환)
predictions = (
    eval(best_result["Predictions"])
    if isinstance(best_result["Predictions"], str)
    else best_result["Predictions"]
)

# 실제 SMP 값
real_smp = [237.61, 151.19, 229.65, 134.87, 134.99, 128.41, 141.13, 115.14]

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(predictions, label="Predictions", marker="o", linestyle="--", color="blue")
plt.plot(real_smp, label="Real SMP", marker="s", linestyle="-", color="red")

# 그래프 설정
plt.title("SMP Prediction vs Real SMP")
plt.xlabel("Year")
plt.ylabel("SMP")
plt.legend()
plt.grid()

# 그래프 저장 (현재 디렉토리에 저장)
plt.savefig("best_lstm_prediction.png")
plt.show()

# 최적 결과 출력
print("Best Result:")
print(best_result)
