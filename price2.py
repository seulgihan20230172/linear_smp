import requests
from lxml import etree
import pandas as pd
from datetime import datetime


current_year = datetime.now().year
previous_year = current_year - 1

url = "https://www.index.go.kr/unity/openApi/xml_stts.do?idntfcId=9010272698CMA0O4&ixCode=5058&statsCode=505801"
response = requests.get(url)

# XML 파싱
root = etree.fromstring(response.content)

data = []

# 모든 '열' 태그를 순회하며 연도와 값 추출
for element in root.xpath("//열"):
    year = element.get("주기")  # 연도
    value = element.text  # 값
    data.append({"Year": year, "Value(%)": value})

# DataFrame으로 변환
price_df = pd.DataFrame(data)


# 각 연도를 4번씩 반복하고 quarter를 추가
price_df = price_df.loc[price_df.index.repeat(4)].reset_index(drop=True)
price_df["Quarter"] = [1, 2, 3, 4] * (len(price_df) // 4)

# 데이터프레임 크기 확인
print(price_df.shape)  # (116, 3)
# DataFrame 출력
print(price_df)
