import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from matplotlib import font_manager, rc, rcParams
import datetime as dt
import sys
import io
import pandas as pd

# -*-coding:utf-8-*-
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# URL을 통해 SDMX 데이터 가져오기
# open_url = "https://bigdata.kepco.co.kr/openapi/v1/commonCode.do?codeTy=cityCd&apiKey=SXpEKyaG3ggPaM293Dlw3CoX7SJBvtFROxy7V2SA"
open_url = "https://www.index.go.kr/unity/openApi/sttsJsonViewer.do?idntfcId=9010272698CMA0O4&ixCode=4221&statsCode=422101"

res = requests.get(open_url)
soup = BeautifulSoup(res.content, "html.parser")
# print(res.status_code)
data = res.json()
# print(type(data))
# print(data)
gni_data = []
current_year = dt.datetime.now().year
# year = int(x.strftime("%Y")) - 1


for item in data:
    if isinstance(item, dict):
        if item.get("항목이름") == "1인당 실질 국민총소득(만 원)":
            year = item.get("시점")
            value = item.get("값")
            gni_data.append({"Year": year, "GNI(만원)": value})

    else:
        print("no dict", type(item))

gni_df = pd.DataFrame(gni_data)


# 각 연도를 4번씩 반복하고 quarter를 추가
gni_df = gni_df.loc[gni_df.index.repeat(4)].reset_index(drop=True)
gni_df["Quarter"] = [1, 2, 3, 4] * (len(gni_df) // 4)

# 데이터프레임 크기 확인
print(gni_df.shape)  # (256, 3)

print(gni_df.columns)
print(gni_df)
