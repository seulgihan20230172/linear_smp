from imp import reload
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
def fetch_profit():
    data = []
    with webdriver.Chrome(service=Service(ChromeDriverManager().install())) as driver:
        driver.get(
            "https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A015760&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701"
        )
        driver.implicitly_wait(10)

        years = []
        for i in range(2, 8):
            year_element = driver.find_element(
                By.XPATH, f'//*[@id="divSonikY"]/table/thead/tr/th[{i}]'
            )
            years.append(year_element.text)

        numbers = []
        for i in range(1, 7):
            number_element = driver.find_element(
                By.XPATH, f'//*[@id="divSonikY"]/table/tbody/tr[13]/td[{i}]'
            )
            numbers.append(number_element.text)

        months = []
        for i in range(0, 6):
            month_element = years[i].split("/")[-1]
            months.append(month_element)

        # 년도와 숫자 출력
        for year, number, month in zip(years, numbers, months):
            data.append(
                {"Year": year.split("/")[0], "Profit(만원)": number, "Month": month}
            )
            # print(f"{year}의 값: {number}")
    df = pd.DataFrame(data)
    return df


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# 전역 변수로 profit_df 정의
profit_df = fetch_profit()

if __name__ == "__main__":
    print(profit_df)
