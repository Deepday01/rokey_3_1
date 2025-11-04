import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import requests
import folium
import warnings
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# NYC 프랜차이즈 상권 분석 및 저평가 지역 탐색 시스템 (최종 버전)
# ROI는 투자금 회수 예상기간(초반투자금/월 순이익)
# ROI 및 전략적 잠재력 점수 분석 -> 둘의 관계를 회귀모델(1차)로 분석 -> 전략적 잠재력 점수 대비 예측 ROI 값 계산 -> 저평가 지역 도출
# 전략적 잠재력이 상위 40% 이상이면서 ROI가 예측보다 높은 지역을 저평가 상권으로 선정
# 선형회귀보다 로그회귀를 이용하여 계산
# 마지막으로 로그회귀를 통해 계산한 모델의 평가 진행
# calculate_income_score(): 가구 소득 기반 점수
# calculate_transit_density(): 교통/픽업 수 기반 점수
# calculate_competition_index(): 경쟁 점수
# calculate_mixed_use(): 상업·주거 혼합 점수
# calculate_subway_access(): 지하철 접근성 점수
class NYCFranchiseAnalyzer:

    def __init__(self):
        self.nta_data = None
        self.weights = {
            'median_income': 0.25,
            'transit_density': 0.20,
            'competition_index': 0.20,
            'mixed_use': 0.15,
            'subway_access': 0.20
        }
        self.financial_params = {
            'initial_investment': 150000,
            'fixed_cost_monthly': 8000,
            'variable_cost_rate': 0.35,
            'target_roi_months': 30
        }

    def fetch_nta_boundaries(self):
        try:
            url = "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson"
            res = requests.get(url, timeout=20)
            geojson = res.json()
            self.nta_data = gpd.GeoDataFrame.from_features(geojson['features'])
            self.nta_data.crs = "EPSG:4326"
            print(f"{len(self.nta_data)}개 NTA 상권 데이터 로드 완료")
        except Exception as e:
            print(f"NYC OpenData 접근 실패 ({e}) → 샘플 데이터 사용")
            self._create_sample_data()

    def _create_sample_data(self):
        sample = [
            {'ntaname': 'Midtown', 'lat': 40.7549, 'lon': -73.9840},
            {'ntaname': 'Chelsea', 'lat': 40.7489, 'lon': -73.9997},
            {'ntaname': 'SoHo', 'lat': 40.7282, 'lon': -74.0021},
            {'ntaname': 'Upper East Side', 'lat': 40.7736, 'lon': -73.9566},
            {'ntaname': 'Lower East Side', 'lat': 40.7154, 'lon': -73.9840},
        ]
        geom = [Point(x['lon'], x['lat']).buffer(0.01) for x in sample]
        self.nta_data = gpd.GeoDataFrame(sample, geometry=geom, crs="EPSG:4326")

    def calculate_income_score(self):
        np.random.seed(42)
        self.nta_data['median_income'] = np.random.normal(75000, 25000, len(self.nta_data)).clip(30000, 150000)
        self.nta_data['income_score'] = (
            (self.nta_data['median_income'] - self.nta_data['median_income'].min()) /
            (self.nta_data['median_income'].max() - self.nta_data['median_income'].min()) * 100
        )

    def calculate_transit_density(self):
        np.random.seed(43)
        self.nta_data['daily_pickups'] = np.random.poisson(5000, len(self.nta_data))
        self.nta_data['transit_score'] = (
            (self.nta_data['daily_pickups'] - self.nta_data['daily_pickups'].min()) /
            (self.nta_data['daily_pickups'].max() - self.nta_data['daily_pickups'].min()) * 100
        )

    def calculate_competition_index(self):
        np.random.seed(44)
        self.nta_data['competitor_count'] = np.random.poisson(25, len(self.nta_data))
        max_c = self.nta_data['competitor_count'].max()
        self.nta_data['competition_score'] = (max_c - self.nta_data['competitor_count']) / max_c * 100

    def calculate_mixed_use(self):
        np.random.seed(45)
        self.nta_data['commercial_ratio'] = np.random.uniform(0.2, 0.8, len(self.nta_data))
        self.nta_data['mixed_use_score'] = 100 - abs(self.nta_data['commercial_ratio'] - 0.5) * 200

    def calculate_subway_access(self):
        np.random.seed(46)
        self.nta_data['subway_ridership'] = np.random.normal(50000, 20000, len(self.nta_data)).clip(10000, 100000)
        self.nta_data['subway_score'] = (
            (self.nta_data['subway_ridership'] - self.nta_data['subway_ridership'].min()) /
            (self.nta_data['subway_ridership'].max() - self.nta_data['subway_ridership'].min()) * 100
        )

    def calculate_strategic_potential(self):
        w = self.weights
        self.nta_data['strategic_potential'] = (
            self.nta_data['income_score'] * w['median_income'] +
            self.nta_data['transit_score'] * w['transit_density'] +
            self.nta_data['competition_score'] * w['competition_index'] +
            self.nta_data['mixed_use_score'] * w['mixed_use'] +
            self.nta_data['subway_score'] * w['subway_access']
        )

    def calculate_financial_risk(self):
        f = self.financial_params
        self.nta_data['estimated_monthly_revenue'] = self.nta_data['strategic_potential'] * 500 + 20000
        self.nta_data['monthly_profit'] = (
            self.nta_data['estimated_monthly_revenue'] * (1 - f['variable_cost_rate']) - f['fixed_cost_monthly']
        )
        self.nta_data['roi_months'] = f['initial_investment'] / self.nta_data['monthly_profit'].clip(lower=1)

    def classify_priorities(self):
        cond = [
            (self.nta_data['strategic_potential'] >= 75) & (self.nta_data['roi_months'] <= 30),
            (self.nta_data['strategic_potential'] >= 75) & (self.nta_data['roi_months'] > 30),
            (self.nta_data['strategic_potential'] < 75) & (self.nta_data['roi_months'] <= 30),
        ]
        val = ['Top Priority', 'High Potential', 'Quick ROI']
        self.nta_data['priority_class'] = np.select(cond, val, default='Monitor')


    def generate_visualizations(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.nta_data, x='strategic_potential', y='roi_months',
                        hue='priority_class', s=120, palette='Set2')
        plt.axvline(75, color='red', ls='--', alpha=0.5)
        plt.axhline(30, color='red', ls='--', alpha=0.5)
        plt.title('Strategic Potential vs ROI Period', fontsize=13)
        plt.xlabel('Strategic Potential Score')
        plt.ylabel('ROI (Months)')
        plt.legend(title='Priority Class')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_interactive_map(self):
        m = folium.Map(location=[40.75, -73.98], zoom_start=11)
        for _, row in self.nta_data.iterrows():
            color = {
                'Top Priority': 'green',
                'High Potential': 'yellow',
                'Quick ROI': 'orange',
                'Monitor': 'red'
            }.get(row['priority_class'], 'gray')

            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=8, color=color, fill=True, fill_opacity=0.6,
                popup=(f"<b>{row['ntaname']}</b><br>"
                       f"Potential: {row['strategic_potential']:.1f}<br>"
                       f"ROI: {row['roi_months']:.1f} mo<br>"
                       f"Class: {row['priority_class']}")
            ).add_to(m)

        m.save("nyc_franchise_interactive_map.html")

def plot_undervalued_on_map(analyzer, undervalued_log):
    # NYC 중심 좌표
    nyc_center = [40.75, -73.98]
    m = folium.Map(location=nyc_center, zoom_start=12)

    # 전체 상권 표시
    for _, row in analyzer.nta_data.iterrows():
        folium.CircleMarker(
            location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            radius=6,
            color='gray',
            fill=True,
            fill_color='gray',
            fill_opacity=0.4,
            popup=folium.Popup(
                f"{row['ntaname']}<br>전략점수: {row['strategic_potential']:.1f}<br>ROI(개월): {row['roi_months']:.1f}",
                max_width=250
            )
        ).add_to(m)

    # 저평가 상권 표시
    for _, row in undervalued_log.iterrows():
        folium.CircleMarker(
            location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            radius=8,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=folium.Popup(
                f"{row['ntaname']}<br>전략점수: {row['strategic_potential']:.1f}<br>ROI(개월): {row['roi_months']:.1f}<br>예측ROI: {row['predicted_roi_log']:.1f}",
                max_width=250
            )
        ).add_to(m)

    return m

if __name__ == "__main__":
    analyzer = NYCFranchiseAnalyzer()
    analyzer.fetch_nta_boundaries()
    analyzer.calculate_income_score()
    analyzer.calculate_transit_density()
    analyzer.calculate_competition_index()
    analyzer.calculate_mixed_use()
    analyzer.calculate_subway_access()
    analyzer.calculate_strategic_potential()
    analyzer.calculate_financial_risk()

    X = analyzer.nta_data[['strategic_potential']].values
    y = np.log(analyzer.nta_data['roi_months'].values)  # 로그 변환

    log_model = LinearRegression()
    log_model.fit(X, y)

    # 예측치 계산 (로그 스케일 → 원래 스케일로 변환)
    analyzer.nta_data['predicted_roi_log'] = np.exp(log_model.predict(X))
    analyzer.nta_data['roi_gap_log'] = analyzer.nta_data['roi_months'] - analyzer.nta_data['predicted_roi_log']

    # 저평가 상권 선택
    threshold_potential = analyzer.nta_data['strategic_potential'].quantile(0.6)
    undervalued_log = analyzer.nta_data[
        (analyzer.nta_data['strategic_potential'] >= threshold_potential) &
        (analyzer.nta_data['roi_gap_log'] > 0)
    ].sort_values('roi_gap_log', ascending=False)

    print(undervalued_log[['ntaname', 'strategic_potential', 'roi_months', 'predicted_roi_log', 'roi_gap_log']])

    plt.figure(figsize=(10,6))

    # 전체 상권
    plt.scatter(analyzer.nta_data['strategic_potential'], analyzer.nta_data['roi_months'],
                c='lightgray', label='전체 상권', alpha=0.6)

    # 로그 회귀 예측선 (정렬 후 선 연결)
    sorted_idx = np.argsort(analyzer.nta_data['strategic_potential'])
    plt.plot(analyzer.nta_data['strategic_potential'].iloc[sorted_idx],
             analyzer.nta_data['predicted_roi_log'].iloc[sorted_idx],
             color='blue', label='로그 회귀 예측선', linewidth=2)

    # 저평가 상권
    plt.scatter(undervalued_log['strategic_potential'], undervalued_log['roi_months'],
                c='red', s=120, label='저평가 상권')

    plt.gca().invert_yaxis()
    plt.title("로그 회귀 기반 상대적 저평가 상권 탐색", fontsize=13)
    plt.xlabel("전략적 잠재력 점수")
    plt.ylabel("ROI (개월)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 결과 CSV 저장
    undervalued_log.to_csv("undervalued_zones_log_regression_utf8.csv", index=False, encoding='utf-8-sig')
    nyc_map = plot_undervalued_on_map(analyzer, undervalued_log)
    nyc_map.save("nyc_undervalued_map.html")

    # 회귀 모델 평가
    # 실제 ROI
    y_true = analyzer.nta_data['roi_months'].values

    # 예측 ROI (로그 회귀)
    y_pred = analyzer.nta_data['predicted_roi_log'].values

    # R²
    r2 = r2_score(y_true, y_pred)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("로그 회귀 모델 평가 지표:")
    print(f"R²: {r2:.3f}") # R²: 0~1 사이, 1에 가까울수록 전략적 잠재력 → ROI 관계를 잘 설명
    print(f"MAE: {mae:.2f}개월")
    print(f"RMSE: {rmse:.2f}개월")

