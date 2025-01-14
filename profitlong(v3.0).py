from bs4 import BeautifulSoup
import requests

# 웹 페이지 URL
url = "https://home.kepco.co.kr/kepco/KE/E/htmlView/KEEBPP0010101.do?menuCd=FN270101"

# 페이지를 요청하여 HTML을 가져옵니다.
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# '이전년도 재무제표' 섹션 찾기
previous_year_section = soup.find("h2", class_="h2tit", string="이전년도 재무제표")

# '이전년도 재무제표' 다음 테이블 찾기
previous_year_table = previous_year_section.find_next("table")

# '손익계산서' 섹션 찾기
income_statement_section = soup.find("td", string="손익계산서")

# '이전년도 재무제표'와 '손익계산서'의 링크를 찾기
income_statement_links = []

# 이전년도 재무제표 링크 추출
for row in previous_year_table.find_all("tr"):
    cols = row.find_all("td")
    if len(cols) > 1:
        # '손익계산서'가 있는 열만 찾기
        income_statement_link = cols[2].find(
            "a", title=lambda x: x and "손익계산서 PDF 다운로드" in x
        )
        if income_statement_link:
            # 링크 앞에 URL을 추가
            full_link = "https://home.kepco.co.kr" + income_statement_link["href"]
            income_statement_links.append(full_link)

# 손익계산서에 해당하는 PDF 링크 추출
pdf_links = []

# '손익계산서'를 포함한 tr 태그를 찾고, 그 안에서 PDF 링크를 추출
for tr_tag in soup.find_all("tr"):
    if "손익계산서" in tr_tag.text:
        # 해당 tr 태그 안에서 PDF 링크 찾기
        for a_tag in tr_tag.find_all("a", class_="ico_pdf"):
            link = a_tag.get("href")
            if link:
                # 링크 앞에 URL을 추가
                full_link = "https://home.kepco.co.kr" + link
                pdf_links.append(full_link)

# 링크들의 총 개수를 세고 출력
total_links = income_statement_links + pdf_links
print(f"전체 PDF 다운로드 링크 총 개수: {len(total_links)}")

# 각 링크 출력
for link in total_links:
    print(link)
