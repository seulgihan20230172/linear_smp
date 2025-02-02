import pandas as pd

smp = {
    2023: 167.11,
    2022: 196.65,
    2021: 94.34,
    2020: 68.87,
    2019: 90.74,
    2018: 95.16,
    2017: 81.77,
    2016: 77.06,
    2015: 101.76,
    2014: 142.26,
    2013: 152.1,
    2012: 160.83,
    2011: 126.63,
    2011: 126.63,
    2010: 117.77,
    2009: 105.08,
    2008: 122.65,
    2007: 83.84,
    2006: 79.27,
    2005: 62.12,
    2004: 55.97,
}
smp_str = {str(year): value for year, value in smp.items()}
smp_df = pd.DataFrame(list(smp_str.items()), columns=["Year", "SMP"])


# 각 연도를 4번씩 반복하고 quarter를 추가
smp_df = smp_df.loc[smp_df.index.repeat(4)].reset_index(drop=True)
smp_df["Quarter"] = [1, 2, 3, 4] * (len(smp_df) // 4)

# 데이터프레임 크기 확인
print(smp_df.shape)  # (80, 3)
print(smp_df)
