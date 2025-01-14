import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from matplotlib import font_manager, rc, rcParams
import datetime as dt

open_url = "https://www.index.go.kr/unity/openApi/sttsJsonViewer.do?idntfcId=9010272698CMA0O4&ixCode=4221&statsCode=422101"

res = requests.get(open_url)
soup = BeautifulSoup(res.content, "html.parser")
print(res.status_code)
data = res.json()
# print(type(data))
# print(data)
x = dt.datetime.now()
year = int(x.strftime("%Y")) - 1
print(year)

for item in data:
    if isinstance(item, dict):
        print(item.get("항목이름"))

    else:
        print("no dict", type(item))
