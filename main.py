import pandas as pd
import numpy as np
from gni import gni_df
from price2 import price_df

# from profit import profit_df
from smp_text import smp_df
from profit_text import profit_df
import sys

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

print(profit_df["Year"].dtype)  # profit_df Year 열 데이터 타입
print(smp_df["Year"].dtype)
print(gni_df["Year"].dtype)  # gni_df Year 열 데이터 타입
print(price_df["Year"].dtype)  # price_df Year

# 각 DataFrame에서 공통된 연도 추출
common_years = set(profit_df["Year"]).intersection(
    gni_df["Year"], price_df["Year"], smp_df["Year"]
)

# 공통된 연도에 해당하는 데이터 추출
common_data = []
print(f"common_years:{common_years}")
for year in common_years:
    data_row = {
        "Year": year,
        "Profit": (
            profit_df[profit_df["Year"] == year]["Profit"].values[0]
            if year in profit_df["Year"].values
            else None
        ),
        "GNI": (
            gni_df[gni_df["Year"] == year]["GNI(만원)"].values[0]
            if year in gni_df["Year"].values
            else None
        ),
        "Price": (
            price_df[price_df["Year"] == year]["Value(%)"].values[0]
            if year in price_df["Year"].values
            else None
        ),
        "SMP": (
            smp_df[smp_df["Year"] == year]["SMP"].values[0]
            if year in price_df["Year"].values
            else None
        ),
    }
    common_data.append(data_row)


# DataFrame으로 변환
common_df = pd.DataFrame(common_data)
common_df = common_df.sort_values(by="Year", ascending=True).reset_index(drop=True)

# NumPy 배열로 변환
numpy_array = common_df.to_numpy()

# 결과 출력
print("Common years and data:")
print(common_df)
print("\nNumPy:")
print(numpy_array)
