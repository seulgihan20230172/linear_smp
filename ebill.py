from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import sys
import io
import pandas as pd

# -*-coding:utf-8-*-
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ebil_list = []

with webdriver.Chrome(service=Service(ChromeDriverManager().install())) as driver:
    driver.get(
        "https://epsis.kpx.or.kr/epsisnew/selectEksaAscAscChart.do?menuId=060501"
    )
    driver.implicitly_wait(10)

    # 용도 리스트
    usage_types = [
        "주택용",
        "일반용",
        "교육용",
        "산업용",
        "농사용",
        "가로등",
        "심야",
        "합계",
    ]

    # 연도 범위 설정 (예: 2015년부터 2023년까지)
    for year in range(2015, 2024):
        # 특정 연도로 이동하는 로직 추가 필요
        # 예: 드롭다운 메뉴에서 연도 선택
        # driver.find_element(...).click()  # 연도 선택 코드 추가 필요

        # 년도 정보 가져오기

        for i in range(0, 22):
            year_element = driver.find_element(
                By.XPATH, f'//*[@id="grid1"]/div/div/div[1]/span[{3+i}]'
            )
            actual_year = year_element.text

            element = driver.find_element(
                By.XPATH, f'//*[@id="rMateH5__Content172"]/span[{33+i}]'
            )  # 값 수정해야함....
            value = element.text

            ebil_list.append(
                {"Year": actual_year, "Type": "일반용", "Data (MWh)": value}
            )

# DataFrame으로 변환
ebill_df = pd.DataFrame(ebil_list)

# 결과 출력
print(ebill_df)
