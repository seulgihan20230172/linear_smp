import os
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

pdf_path = [
    "C:\\Users\\hanse\\Downloads\\2.연결재무제표_나.연결포괄손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_15.4Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\2015.3Q_연결_손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_15.2Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\2. 2015.1Q_연결_손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_16.4Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_16.3Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_16.2Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_16.1Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_17.4Q포괄손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_17.3Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_17.2Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_17.1Q손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\18.4Q_연결_PL.pdf",
    "C:\\Users\\hanse\\Downloads\\2018년 3분기 손익계산서 PDF(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\18.2Q_연결_PL.pdf",
    "C:\\Users\\hanse\\Downloads\\연결_포괄손익계산서_18.1Q.pdf",
    "C:\\Users\\hanse\\Downloads\\19.4Q_PL(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\19.3Q_PL(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\19.2Q_PL(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\19.1Q_PL(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\2020년_3분기_연결_2_손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\20.2Q_PL(연결).pdf",
    "C:\\Users\\hanse\\Downloads\\20.1Q_연결_손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\연결PL.pdf",
    "C:\\Users\\hanse\\Downloads\\21.3Q_손익계산서_연결.pdf",
    "C:\\Users\\hanse\\Downloads\\21.2Q_손익계산서_연결.pdf",
    "C:\\Users\\hanse\\Downloads\\21.1Q_손익계산서_연결.pdf",
    "C:\\Users\\hanse\\Downloads\\22.4Q_연결_포괄손익계산서.pdf",
    "C:\\Users\\hanse\\Downloads\\22.3Q_손익계산서_연결.pdf",
    "C:\\Users\\hanse\\Downloads\\22.2Q_손익계산서_연결.pdf",
    "C:\\Users\\hanse\\Downloads\\22.1Q_손익계산서_연결 (6).pdf",
    "C:\\Users\\hanse\\Downloads\\SSAP 2024 Application HAN SEULGI(한슬기).pdf",
    "C:\\Users\\hanse\\Downloads\\Assignment#7-due 20241120.pdf",
    "C:\\Users\\hanse\\Downloads\\Assignment#6-due 20241113.pdf",
    "C:\\Users\\hanse\\Downloads\\한슬기-신청서_20241111.pdf",
    "C:\\Users\\hanse\\Downloads\\IEEE Template_final (1).pdf",
    "C:\\Users\\hanse\\Downloads\\34,26 (2).pdf",
    "C:\\Users\\hanse\\Downloads\\22.1Q_손익계산서_연결 (5).pdf",
    "C:\\Users\\hanse\\Downloads\\22.1Q_손익계산서_연결 (4).pdf",
    "C:\\Users\\hanse\\Downloads\\22.1Q_손익계산서_연결 (3).pdf",
    "C:\\Users\\hanse\\Downloads\\HW#1and2-1.pdf",
    "C:\\Users\\hanse\\Downloads\\IEEE Template_final.pdf",
    "C:\\Users\\hanse\\Downloads\\ESP Intermediate Writing 10-2 - SWW - Lit Review Organization and Outlining.pdf",
    "C:\\Users\\hanse\\Downloads\\34,26 (1).pdf",
    "C:\\Users\\hanse\\Downloads\\34,26.pdf",
    "C:\\Users\\hanse\\Downloads\\P3 - Literature Review Peer Review Worksheet (1).pdf",
    "C:\\Users\\hanse\\Downloads\\P3 - Literature Review Peer Review Worksheet.pdf",
    "C:\\Users\\hanse\\Downloads\\IEEE-Editorial-Style-Manual-for-Authors.pdf",
]
for i in range(0, 49):
    pdf_name = os.path.basename(pdf_path[i])
    try:
        period = re.search(r"\d{2}\.\d", pdf_name).group(
            0
        )  # pdf_name에서 '22.1' 형식 추출
    except:
        period = None
    print(f"{pdf_name} : {period}")
