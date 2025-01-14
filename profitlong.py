import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from pdf2image import convert_from_path
import pytesseract
import sys
import io
import pandas as pd
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# PDF 다운로드 경로 설정
download_dir = r"C:\Users\hanse\Downloads"  # 다운로드 경로 설정
pdf_name = "22.1Q_손익계산서_연결.pdf"
pdf_path = os.path.join(download_dir, pdf_name)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def check_existing_pdf():
    for file in os.listdir(download_dir):
        if file.startswith(pdf_name):
            return os.path.join(download_dir, file)
    return None


# 1. Selenium을 사용하여 PDF 자동 다운로드
def download_pdf():
    # Chrome 브라우저 드라이버 설정
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_dir}
    options.add_experimental_option("prefs", prefs)

    # 브라우저 실행
    driver = webdriver.Chrome(options=options)
    driver.get(
        "https://home.kepco.co.kr/kepco/KE/E/htmlView/KEEBPP0010101.do?menuCd=FN270101"
    )

    # 페이지에서 PDF 다운로드 버튼 클릭 (버튼 위치를 알아내서 클릭해야 함)
    try:
        download_button = driver.find_element(
            By.XPATH,
            '//*[@id="content"]/div[4]/div[2]/table/tbody/tr[2]/td[2]/div/div/a/span/strong',
        )  # 버튼의 실제 XPATH 설정 필요
        download_button.click()
        time.sleep(5)  # 다운로드 대기 시간 설정
    finally:
        driver.quit()


# 2. PDF 파일 텍스트 추출
def extract_text_from_pdf(pdf_path):
    if os.path.exists(pdf_path):
        images = convert_from_path(pdf_path)
        text = ""

        for image in images:
            text += pytesseract.image_to_string(image, lang="kor")
        return text
    else:
        print("PDF 파일이 존재하지 않습니다.")
        return None


# 3. 텍스트에서 단어 검색 및 특정 부분 추출
def find_word_in_text_and_save_to_df(text, word, pdf_name):
    # pdf_name에서 '22.1' 부분만 추출
    period = re.search(r"\d{2}\.\d", pdf_name).group(0)

    # 찾고자 하는 단어가 포함된 줄만 추출
    lines = text.splitlines()
    found_lines = [line for line in lines if word in line]

    # 추출한 데이터를 저장할 리스트
    extracted_data = []

    # 각 줄에서 단어 다음의 숫자 2개 추출
    for line in found_lines:
        # 단어 이후 텍스트를 잘라서 숫자 2개만 추출
        # print(line.split(word, 1))
        right_side = line.split(word, 1)[-1]
        # print("right_side")
        # print(right_side)
        a, b = right_side.split()
        extracted_data.append([period, a, b])

    # DataFrame 생성, 열 이름 설정
    df = pd.DataFrame(extracted_data, columns=["날짜", "금액(전)", "금액(후)"])

    # 결과 출력
    print(df)


# 5. PDF가 이미 존재하는지 확인 후, 존재하지 않으면 다운로드
pdf_path = check_existing_pdf()
if not pdf_path:
    download_pdf()
    pdf_path = check_existing_pdf()

# PDF 파일 텍스트 추출 및 단어 검색
if pdf_path:
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text:
        find_word_in_text_and_save_to_df(pdf_text, "매출총이익(손실)", pdf_name)
    else:
        print("text가 추출되지 않았습니다.")
else:
    print("PDF파일을 찾을 수 없습니다.")
