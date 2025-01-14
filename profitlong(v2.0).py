# xpath불규칙성 반복해결
# 파일명 불규칙성->다운하고 파일이름 search
# 같은 파일만 반복해서 다운됨
# 계속 열고 다운 다시 열고 다운으로 비효율적임->selenium말고 다른걸 생각해야할지도 beautiful soup?

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re

# Set paths and Tesseract configuration
download_dir = r"C:\Users\hanse\Downloads"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Function to check for existing PDF
def check_existing_pdf(file_name):
    for file in os.listdir(download_dir):
        if file.startswith(file_name):
            return os.path.join(download_dir, file)
    return None


# Function to download PDF for a given year and quarter
def download_pdf(year, quarter):
    file_name = f"{year}.{quarter}Q_손익계산서_연결.pdf"
    pdf_path = check_existing_pdf(file_name)

    if pdf_path:
        return pdf_path  # PDF already exists

    # Set up Chrome driver options
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_dir}
    options.add_experimental_option("prefs", prefs)

    # Open browser and navigate to download page
    driver = webdriver.Chrome(options=options)
    driver.get(
        "https://home.kepco.co.kr/kepco/KE/E/htmlView/KEEBPP0010101.do?menuCd=FN270101"
    )

    # Dynamically find and click download button
    try:
        xpath = f'//*[@id="content"]/div[4]/div[2]/table/tbody/tr[2]/td[{quarter+1}]/div/div/a/span/strong'
        """
        //*[@id="content"]/div[4]/div[2]/table/tbody/tr[2]/td[2]/div/div/a/span/strong
        //*[@id="content"]/div[4]/div[2]/table/tbody/tr[2]/td[3]/div/div/a/span/strong
        //*[@id="content"]/div[4]/div[2]/table/tbody/tr[2]/td[4]/div/div/a/span/strong

        //*[@id="content"]/div[5]/div[2]/table/tbody/tr[2]/td[2]/div/div/a/span/strong

        //*[@id="content"]/div[14]/div/table/tbody/tr[1]/td[3]/div/div/a/span/strong
        //*[@id="content"]/div[14]/div/table/tbody/tr[2]/td[3]/div/div/a/span/strong
        //*[@id="content"]/div[14]/div/table/tbody/tr[3]/td[3]/div/div/a/span/strong
        """
        download_button = driver.find_element(By.XPATH, xpath)
        download_button.click()
        time.sleep(5)
    finally:
        driver.quit()

    return check_existing_pdf(file_name)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    if os.path.exists(pdf_path):
        images = convert_from_path(pdf_path)
        text = ""

        for image in images:
            text += pytesseract.image_to_string(image, lang="kor")
        return text
    else:
        print("PDF file does not exist.")
        return None


# Function to search for a word in text and save data to a DataFrame
def find_word_in_text_and_save_to_df(text, word, pdf_name):
    period = re.search(r"\d{2}\.\d", pdf_name).group(0)
    lines = text.splitlines()
    found_lines = [line for line in lines if word in line]
    extracted_data = []

    for line in found_lines:
        right_side = line.split(word, 1)[-1]
        numbers = re.findall(r"\d+", right_side)
        if len(numbers) >= 2:
            a, b = numbers[:2]
            extracted_data.append([period, a, b])

    df = pd.DataFrame(extracted_data, columns=["날짜", "금액(전)", "금액(후)"])
    print(df)
    return df


# Main loop to download, extract text, and process data for each year and quarter
all_dataframes = []
for year in range(2005, 2023):
    if year < 2013:
        pdf_name = f"{year}.손익계산서_연결.pdf"
        pdf_path = download_pdf(year, 1)
        if pdf_path:
            text = extract_text_from_pdf(pdf_path)
            if text:
                df = find_word_in_text_and_save_to_df(
                    text, "매출총이익(손실)", pdf_name
                )
                all_dataframes.append(df)
    else:
        for quarter in range(1, 5):
            pdf_name = f"{year}.{quarter}Q_손익계산서_연결.pdf"
            pdf_path = download_pdf(year, quarter)
            if pdf_path:
                text = extract_text_from_pdf(pdf_path)
                if text:
                    df = find_word_in_text_and_save_to_df(
                        text, "매출총이익(손실)", pdf_name
                    )
                    all_dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame if desired
final_df = pd.concat(all_dataframes, ignore_index=True)
print(final_df)
