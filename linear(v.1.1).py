import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from main import common_df

# 데이터 정리 및 변환
common_df["Profit"] = common_df["Profit"].astype(str).str.replace(",", "", regex=False)
common_df["Profit"] = pd.to_numeric(common_df["Profit"], errors="coerce")
common_df["GNI"] = pd.to_numeric(common_df["GNI"], errors="coerce")
common_df["Price"] = pd.to_numeric(common_df["Price"], errors="coerce")
common_df["SMP"] = pd.to_numeric(common_df["SMP"], errors="coerce")

# 결측치 처리
common_df = common_df.dropna()

# 시각화 및 MSE 계산을 위한 변수 목록
variables = ["Profit", "GNI", "Price"]
results = {}
optimal_degrees = {}

# 반복문: 각 변수별로 분석
for var in variables:
    # 독립 변수와 종속 변수 준비
    X = common_df[[var]].values  # 독립 변수로 Profit, GNI, Price 중 하나 사용
    y_smp = common_df["SMP"].values  # 종속 변수: SMP

    # 선형 회귀
    lin_reg = LinearRegression()
    lin_reg.fit(X, y_smp)
    y_pred_lin = lin_reg.predict(X)

    # 다항 회귀: 최적의 degree 선택 (교차 검증 사용)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees = range(1, 21)  # 테스트할 degree 범위
    mean_cv_scores = []

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        poly_reg = LinearRegression()
        cv_scores = cross_val_score(
            poly_reg, X_poly, y_smp, cv=kf, scoring="neg_mean_squared_error"
        )
        mean_cv_scores.append(-np.mean(cv_scores))

    # 최적 degree 선택
    optimal_degree = degrees[np.argmin(mean_cv_scores)]
    optimal_degrees[var] = optimal_degree

    # 최적 degree를 사용한 다항 회귀
    poly_features = PolynomialFeatures(degree=optimal_degree)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y_smp)
    y_pred_poly = poly_reg.predict(X_poly)

    # MSE 계산
    mse_lin = mean_squared_error(y_smp, y_pred_lin)
    mse_poly = mean_squared_error(y_smp, y_pred_poly)

    # 결과 저장
    results[var] = {
        "Linear MSE": mse_lin,
        "Polynomial MSE": mse_poly,
        "Optimal Degree": optimal_degree,
    }

    # 시각화
    plt.figure(figsize=(16, 9))
    plt.scatter(X.flatten(), y_smp, color="red", label={var})
    plt.plot(X.flatten(), y_pred_lin, color="blue", label="Linear Regression")
    plt.plot(
        X.flatten(),
        y_pred_poly,
        color="green",
        label=f"Polynomial Regression (degree={optimal_degree})",
    )
    plt.title(f"SMP Regression Analysis for {var}")
    plt.xlabel(var)
    plt.ylabel("SMP")
    plt.legend()
    plt.show()

# MSE 및 최적 degree 출력
for var, mse_values in results.items():
    print(f"{var}:")
    for reg_type, mse in mse_values.items():
        if reg_type != "Optimal Degree":
            print(f"  {reg_type}: {mse:.4f}")
        else:
            print(f"  {reg_type}: {mse}")
