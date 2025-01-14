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
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
def check_existing_pdf():
    for file in os.listdir(download_dir):
        if file.startswith(pdf_name):
            return os.path.join(download_dir, file)
    return None
"""


# 다운로드 폴더에서 가장 최근 파일 49개 반환
def get_latest_files_from_download_dir():
    files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
    pdf_files = [f for f in files if f.endswith(".pdf")]
    if not pdf_files:
        return []
    # 파일을 생성 시간 기준으로 정렬 후 상위 49개 반환
    sorted_files = sorted(pdf_files, key=os.path.getctime, reverse=True)
    return sorted_files[:42]


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

    try:
        # div[4~13] 반복, td[2~5] 반복
        for div in range(4, 12):  # div[4]부터 div[13]까지
            for td in range(2, 6):  # td[2]부터 td[5]까지
                xpath = f'//*[@id="content"]/div[{div}]/div[2]/table/tbody/tr[2]/td[{td}]/div/div/a/span/strong'
                download_button = driver.find_element(By.XPATH, xpath)

                # 요소로 스크롤 이동
                driver.execute_script(
                    "arguments[0].scrollIntoView(true);", download_button
                )
                time.sleep(1)  # 스크롤 후 대기

                # JavaScript로 클릭
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(5)  # 파일 다운로드 대기

        # div[12~13] 반복, div[2]로 처리
        for div in range(12, 14):  # div[12]부터 div[13]까지
            for td in range(2, 6):  # td[2]부터 td[5]까지
                xpath = f'//*[@id="content"]/div[{div}]/div/table/tbody/tr[2]/td[{td}]/div/div/a/span/strong'
                download_button = driver.find_element(By.XPATH, xpath)

                # 요소로 스크롤 이동
                driver.execute_script(
                    "arguments[0].scrollIntoView(true);", download_button
                )
                time.sleep(1)  # 스크롤 후 대기

                # JavaScript로 클릭
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(5)  # 파일 다운로드 대기

        xpath = f'//*[@id="content"]/div[12]/div/table/tbody/tr[2]/td[5]/div/div/a/span/strong'
        download_button = driver.find_element(By.XPATH, xpath)

        # 요소로 스크롤 이동
        driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
        time.sleep(1)  # 스크롤 후 대기

        # JavaScript로 클릭
        driver.execute_script("arguments[0].click();", download_button)
        time.sleep(5)  # 파일 다운로드 대기

        """
        # div[14] 반복, tr[1~8] 반복, td[3]
        for tr in range(1, 9):  # tr[1]부터 tr[8]까지
            xpath = f'//*[@id="content"]/div[14]/div/table/tbody/tr[{tr}]/td[3]/div/div/a/span/strong'
            download_button = driver.find_element(By.XPATH, xpath)

            # 요소로 스크롤 이동
            driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
            time.sleep(1)  # 스크롤 후 대기

            # JavaScript로 클릭
            driver.execute_script("arguments[0].click();", download_button)
            time.sleep(5)  # 파일 다운로드 대기
        """
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


"""
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
"""


def find_word_in_text_and_return_data(text, word, pdf_name):

    # 추출한 데이터를 저장할 리스트
    extracted_data = []

    try:
        period = re.search(r"\d{2}\.\d", pdf_name).group(
            0
        )  # pdf_name에서 '22.1' 형식 추출
    except:
        period = pdf_name

    lines = text.splitlines()
    found_lines = [line for line in lines if word in line]

    # 각 줄에서 단어 다음의 숫자 2개 추출
    for line in found_lines:
        try:
            right_side = line.split(word, 1)[-1]
            # 1. 띄어쓰기로 나눔
            parts = right_side.split()[:2]
            # 결과: ['7,/11,931.123', '3,442/156.476']

            # 2. 각 부분에서 '.'와 '/' 제거
            clean_parts = [
                part.replace(".", "").replace("/", "").replace("|", "")
                for part in parts
            ]
            extracted_data.append([period, clean_parts[0], clean_parts[1]])
        except:
            print(f"단어찾기불가: {pdf_name}")

    return extracted_data  # 데이터를 반환


download_pdf()
# 최신 파일 49개 가져오기
pdf_paths = get_latest_files_from_download_dir()

# PDF 파일이 있는지 확인
if pdf_paths:
    print(f"pdf_paths: {pdf_paths},pdf개수: {len(pdf_paths)}")
    all_data = []  # 데이터를 저장할 리스트

    for pdf_path in pdf_paths:
        # PDF 텍스트 추출
        try:
            pdf_text = extract_text_from_pdf(pdf_path)
        except:
            pdf_text = None

        if pdf_text:
            # 파일명에서 날짜 정보 추출
            pdf_name = os.path.basename(pdf_path)
            extracted_data = find_word_in_text_and_return_data(
                pdf_text, "매출총이익", pdf_name
            )  # 데이터 반환

            if extracted_data:
                all_data.extend(extracted_data)  # 결과 리스트에 추가
                print(f"{pdf_name}에서 데이터 추출 개수: {extracted_data}")
            else:
                print(f"매출총이익(손실)을 못찾았습니다: {pdf_name}")

        else:
            print(f"Text가 추출되지 않았습니다. 파일: {pdf_path}")

    # 리스트를 DataFrame으로 변환
    if all_data:
        df = pd.DataFrame(all_data, columns=["날짜", "금액(전)", "금액(후)"])
        df = df.drop_duplicates()  # 중복 제거
        print(df)
    else:
        print("추출된 데이터가 없습니다.")
else:
    print("PDF 파일을 찾을 수 없습니다.")
