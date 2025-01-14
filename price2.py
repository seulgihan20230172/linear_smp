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

# DataFrame 출력
print(price_df)
