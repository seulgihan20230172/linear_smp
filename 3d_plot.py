import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 한글 폰트 설정 (예: 나눔고딕)
font_path = r"C:\Users\hanse\Downloads\nanum-gothic\NanumGothicLight.ttf"
font_prop = fm.FontProperties(fname=font_path, size=12)

# 데이터 생성
prices = np.random.randint(100, 1000, size=20)  # 가격 데이터
years = np.random.choice([2021, 2022, 2023], size=20)  # 연도 데이터
categories = np.random.choice(
    ["주택용", "일반용", "교육용", "산업용", "농사용"], size=20
)  # 카테고리 데이터

# 카테고리를 정수로 변환
category_mapping = {"주택용": 1, "일반용": 2, "교육용": 3, "산업용": 4, "농사용": 5}
z = [category_mapping[category] for category in categories]  # 정수로 변환된 z축 데이터

# figure 크기 설정
fig = plt.figure(figsize=(8, 6))

# 3D axes 생성
ax = fig.add_subplot(111, projection="3d")

# scatter() 함수에 준비된 x, y, z 배열 값을 입력
ax.scatter(prices, years, z, marker="o", s=50, c="darkgreen")

# 축 레이블 설정
ax.set_xlabel("가격", fontproperties=font_prop)
ax.set_ylabel("연도", fontproperties=font_prop)
ax.set_zlabel("카테고리", fontproperties=font_prop)

# 카테고리 이름 설정
ax.set_zticks(list(category_mapping.values()))
ax.set_zticklabels(list(category_mapping.keys()), fontproperties=font_prop)

plt.show()
