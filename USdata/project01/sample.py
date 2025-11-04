import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1. 데이터 수집 ----
years = [2015, 2023]
base_url = "https://api.census.gov/data/{}/acs/acs1"
all_data = []

for year in years:
    url = base_url.format(year)
    params = {
        "get": "NAME,B01003_001E,B19013_001E",
        "for": "state:*"
    }
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    all_data.append(df)

df_all = pd.concat(all_data)
df_all["B01003_001E"] = df_all["B01003_001E"].astype(float)
df_all["B19013_001E"] = df_all["B19013_001E"].astype(float)

# ---- 2. 증가율 계산 ----
df_2015 = df_all[df_all["year"] == 2015][["NAME", "B01003_001E", "B19013_001E"]].set_index("NAME")
df_2023 = df_all[df_all["year"] == 2023][["NAME", "B01003_001E", "B19013_001E"]].set_index("NAME")

growth = pd.DataFrame()
growth["Population Growth (%)"] = ((df_2023["B01003_001E"] - df_2015["B01003_001E"]) / df_2015["B01003_001E"]) * 100
growth["Income Growth (%)"] = ((df_2023["B19013_001E"] - df_2015["B19013_001E"]) / df_2015["B19013_001E"]) * 100
growth = growth.reset_index().rename(columns={"NAME": "State"})

# ---- 3. 상위/하위 10개 주 ----
top_pop = growth.sort_values("Population Growth (%)", ascending=False).head(10)
top_income = growth.sort_values("Income Growth (%)", ascending=False).head(10)

# ---- 4. 그래프 스타일 설정 ----
plt.style.use("seaborn-v0_8-whitegrid")

# ---- (A) 인구 증가율 TOP 10 ----
plt.figure(figsize=(9,6))
plt.barh(top_pop["State"], top_pop["Population Growth (%)"], color="skyblue")
plt.gca().invert_yaxis()  # 위에서 아래로 순위 표시
plt.title("U.S. Population Growth Top 10 States (2015–2023)", fontsize=14)
plt.xlabel("Population Growth (%)")
plt.tight_layout()
plt.show()

# ---- (B) 소득 증가율 TOP 10 ----
plt.figure(figsize=(9,6))
plt.barh(top_income["State"], top_income["Income Growth (%)"], color="lightgreen")
plt.gca().invert_yaxis()
plt.title(" U.S. Median Income Growth Top 10 States (2015–2023)", fontsize=14)
plt.xlabel("Income Growth (%)")
plt.tight_layout()
plt.show()