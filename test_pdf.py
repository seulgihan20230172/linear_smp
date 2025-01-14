import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import io
import pandas as pd
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# PDF 저장 경로 설정
download_dir = r"C:\Users\hanse\Downloads"  # PDF를 저장할 경로
os.makedirs(download_dir, exist_ok=True)


# PDF 다운로드 함수
def download_pdf_from_url(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP 요청 에러 체크
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"다운로드 성공: {save_path}")
    except Exception as e:
        print(f"다운로드 실패: {url}, 오류: {e}")


# Selenium으로 특정 PDF 링크 수집 및 다운로드
def collect_and_download_single_pdf():
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    driver.get(
        "https://home.kepco.co.kr/kepco/KE/E/htmlView/KEEBPP0010101.do?menuCd=FN270101"
    )

    try:
        # iframe 전환 (필요시 실행)
        iframe_elements = driver.find_elements(By.TAG_NAME, "iframe")
        if iframe_elements:
            driver.switch_to.frame(iframe_elements[0])  # 첫 번째 iframe으로 전환

        # 요소가 로드될 때까지 대기
        xpath = '//*[@id="content"]/div[12]/div/table/tbody/tr[2]/td[2]/div/div/a'
        link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )

        # PDF URL 가져오기
        pdf_url = link_element.get_attribute("href")
        print(f"PDF 링크: {pdf_url}")

        # PDF 다운로드
        if pdf_url:
            pdf_name = "single_file.pdf"
            save_path = os.path.join(download_dir, pdf_name)
            download_pdf_from_url(pdf_url, save_path)
        else:
            print("PDF 링크를 찾을 수 없습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        driver.quit()


# 실행
collect_and_download_single_pdf()
