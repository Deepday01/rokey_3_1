import pandas as pd
import matplotlib.pyplot as plt

# 1️ 데이터 불러오기

df = pd.read_csv(r"C:/rokey/bughunters/project01/cbp23st.txt", quotechar='"', dtype=str)
df.columns = df.columns.str.strip().str.replace('"', '').str.upper()

# 2️ 숫자형 변환
for col in ["EMP", "QP1", "AP", "EST"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3️ FIPS 코드 → 주 이름 매핑
fips_to_state = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
    "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming"
}

df["STATE_NAME"] = df["FIPSTATE"].map(fips_to_state)

df["NAICS_2"] = df["NAICS"].str[:2]  # 상위 2자리
naics_2digit_map = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management and Remediation Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration"
}
df["INDUSTRY_NAME"] = df["NAICS_2"].map(naics_2digit_map)

# 4️ 주별 요약
state_summary = (
    df.groupby("STATE_NAME")[["EMP", "AP", "EST"]]
    .sum()
    .sort_values("EMP", ascending=False)
)

state_summary["AVG_PAY_PER_EMP"] = state_summary["AP"] / state_summary["EMP"]

# 5️ 산업별 요약
industry_summary = df.groupby("INDUSTRY_NAME")[["EMP", "AP", "EST"]].sum()
industry_summary["AVG_PAY_PER_EMP"] = industry_summary["AP"] / industry_summary["EMP"]

industry_summary["AVG_PAY_PER_EMP"] = industry_summary["AP"] / industry_summary["EMP"]

# 6️ 시각화 - 주별 고용 상위 10
plt.figure(figsize=(10,6))
top_states = state_summary.sort_values("EMP", ascending=False).head(10)
plt.barh(top_states.index, top_states["EMP"], color="seagreen")
plt.gca().invert_yaxis()
plt.title("Top 10 States by Employment")
plt.xlabel("Employment")
plt.tight_layout()
plt.show()

# 7️ 시각화 - 산업별 평균 급여 상위 10
plt.figure(figsize=(10,6))
top_industries = industry_summary.sort_values("AVG_PAY_PER_EMP", ascending=False).head(10)
plt.barh(top_industries.index, top_industries["AVG_PAY_PER_EMP"], color="cornflowerblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Industries by Average Payroll per Employee")
plt.xlabel("Average Payroll per Employee ($)")
plt.tight_layout()
plt.show()

# 8️ 주요 통계 출력
print("\n [Top 10 States by Employment]")
print(state_summary.head(10))

print("\n [Top 10 Industries by Average Annual Payroll per Employee]")
print(top_industries[["EMP", "AVG_PAY_PER_EMP"]])