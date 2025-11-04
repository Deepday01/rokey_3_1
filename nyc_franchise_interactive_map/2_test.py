"""
NYC 프랜차이즈 출점 전략 분석 시스템
Strategic Location Analysis for Franchise Expansion in New York City

Author: Bughunters
Date: October 2025
Purpose: 뉴욕시 상권 분석을 통한 최적 출점 위치 도출
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import requests
import json
from datetime import datetime
import folium
from folium import plugins # 지도
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NYCFranchiseAnalyzer:
    """뉴욕시 프랜차이즈 출점 전략 분석 클래스"""
    
    def __init__(self):
        """분석 시스템 초기화"""
        self.nta_data = None
        self.analysis_df = None
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
        """NTA 경계 데이터 로드"""
        print("📍 Step 1: NTA 경계 데이터 로딩...")
        
        nta_url = "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson"
        
        try:
            response = requests.get(nta_url, timeout=30)
            nta_geojson = response.json()
            self.nta_data = gpd.GeoDataFrame.from_features(nta_geojson['features'])
            self.nta_data.crs = "EPSG:4326"
            
            if 'ntaname' not in self.nta_data.columns and 'nta_name' in self.nta_data.columns:
                self.nta_data['ntaname'] = self.nta_data['nta_name']
            
            print(f"✓ {len(self.nta_data)} 개 상권(NTA) 경계 데이터 로드 완료")
            return True
        except Exception as e:
            print(f"✗ NTA 데이터 로드 실패: {e}")
            self._create_sample_nta_data()
            return False
    
    def _create_sample_nta_data(self):
        """샘플 NTA 데이터 생성"""
        print("⚠ 샘플 데이터로 분석을 진행합니다...")
        # 맨해튼 주요 상권 샘플
        sample_areas = [
            {'ntaname': 'Midtown-Midtown South', 'lat': 40.7549, 'lon': -73.9840},
            {'ntaname': 'Upper East Side', 'lat': 40.7736, 'lon': -73.9566},
            {'ntaname': 'Upper West Side', 'lat': 40.7870, 'lon': -73.9754},
            {'ntaname': 'Chelsea-Clinton', 'lat': 40.7489, 'lon': -73.9997},
            {'ntaname': 'Greenwich Village-SoHo', 'lat': 40.7282, 'lon': -74.0021},
            {'ntaname': 'Lower East Side', 'lat': 40.7154, 'lon': -73.9840},
            {'ntaname': 'Financial District', 'lat': 40.7074, 'lon': -74.0113},
            {'ntaname': 'Murray Hill-Gramercy', 'lat': 40.7450, 'lon': -73.9808},
            {'ntaname': 'East Village', 'lat': 40.7264, 'lon': -73.9818},
            {'ntaname': 'Tribeca', 'lat': 40.7163, 'lon': -74.0086}
        ]
        
        geometries = [Point(area['lon'], area['lat']).buffer(0.01) for area in sample_areas]
        self.nta_data = gpd.GeoDataFrame(
            sample_areas,
            geometry=geometries,
            crs="EPSG:4326"
        )
    
    def calculate_income_score(self):
        """KSF 1: 중위 가구 소득 점수 계산"""
        print("💰 Step 2: 중위 가구 소득 분석...")

        # NYC Census Data - 샘플 데이터 (실제로는 ACS API 사용)
        # URL: https://data.cityofnewyork.us/resource/5ebf-z9qf.json
        
        np.random.seed(42)
        self.nta_data['median_income'] = np.random.normal(75000, 25000, len(self.nta_data))
        self.nta_data['median_income'] = self.nta_data['median_income'].clip(30000, 150000)
        
        min_income = self.nta_data['median_income'].min()
        max_income = self.nta_data['median_income'].max()
        self.nta_data['income_score'] = (
            (self.nta_data['median_income'] - min_income) / (max_income - min_income) * 100
        )
        
        print(f"✓ 소득 분석 완료 (평균: ${self.nta_data['median_income'].mean():,.0f})")
    
    def calculate_transit_density(self):
        """KSF 2: 교통 밀도 점수"""
        print("🚕 Step 3: 교통 밀도 분석...")

        # TLC Trip Records - 샘플 데이터
        # URL: https://data.cityofnewyork.us/resource/t29m-gskq.json        
        
        np.random.seed(43)
        self.nta_data['daily_pickups'] = np.random.poisson(5000, len(self.nta_data))
        
        min_trips = self.nta_data['daily_pickups'].min()
        max_trips = self.nta_data['daily_pickups'].max()
        self.nta_data['transit_score'] = (
            (self.nta_data['daily_pickups'] - min_trips) / (max_trips - min_trips) * 100
        )
        
        print(f"✓ 교통 밀도 분석 완료 (평균 일일 픽업: {self.nta_data['daily_pickups'].mean():.0f})")
    
    def calculate_competition_index(self):
        """KSF 3: 경쟁사 지수"""
        print("☕ Step 4: 경쟁 환경 분석...")

        # DCA Licenses - Active Businesses
        # URL: https://data.cityofnewyork.us/resource/w7w3-xahh.json

        np.random.seed(44)
        self.nta_data['competitor_count'] = np.random.poisson(25, len(self.nta_data))
        
        max_comp = self.nta_data['competitor_count'].max()
        self.nta_data['competition_score'] = (
            (max_comp - self.nta_data['competitor_count']) / max_comp * 100
        )
        
        print(f"✓ 경쟁 분석 완료 (평균 경쟁사 수: {self.nta_data['competitor_count'].mean():.1f})")
    
    def calculate_mixed_use(self):
        """KSF 4: 주거/상업 혼합 비율"""
        print("🏢 Step 5: 토지 이용 분석...")
        
        # PLUTO Data - Land Use
        # URL: https://data.cityofnewyork.us/resource/64uk-42ks.json

        np.random.seed(45)
        self.nta_data['commercial_ratio'] = np.random.uniform(0.2, 0.8, len(self.nta_data))
        
        self.nta_data['mixed_use_score'] = (
            100 - abs(self.nta_data['commercial_ratio'] - 0.5) * 200
        )
        
        print(f"✓ 토지 이용 분석 완료 (평균 상업 비율: {self.nta_data['commercial_ratio'].mean():.1%})")
    
    def calculate_subway_access(self):
        """KSF 5: 지하철 접근성"""
        print("🚇 Step 6: 지하철 접근성 분석...")

        # MTA Turnstile Data
        # URL: https://data.cityofnewyork.us/resource/wujg-7c2s.json

        np.random.seed(46)
        self.nta_data['subway_ridership'] = np.random.normal(50000, 20000, len(self.nta_data))
        self.nta_data['subway_ridership'] = self.nta_data['subway_ridership'].clip(10000, 100000)
        
        min_rider = self.nta_data['subway_ridership'].min()
        max_rider = self.nta_data['subway_ridership'].max()
        self.nta_data['subway_score'] = (
            (self.nta_data['subway_ridership'] - min_rider) / (max_rider - min_rider) * 100
        )
        
        print(f"✓ 지하철 분석 완료 (평균 일일 승하차: {self.nta_data['subway_ridership'].mean():,.0f})")
    
    def calculate_strategic_potential(self):
        """전략적 잠재력 종합 점수 계산"""
        print("\n📊 Step 7: 전략적 잠재력 종합 평가...")
        
        self.nta_data['strategic_potential'] = (
            self.nta_data['income_score'] * self.weights['median_income'] +
            self.nta_data['transit_score'] * self.weights['transit_density'] +
            self.nta_data['competition_score'] * self.weights['competition_index'] +
            self.nta_data['mixed_use_score'] * self.weights['mixed_use'] +
            self.nta_data['subway_score'] * self.weights['subway_access']
        )
        
        print(f"✓ 전략적 잠재력 점수 계산 완료 (평균: {self.nta_data['strategic_potential'].mean():.1f})")
    
    def calculate_financial_risk(self):
        """재무적 리스크 평가"""
        print("💵 Step 8: 재무적 리스크 분석...")
        
        self.nta_data['estimated_monthly_revenue'] = (
            self.nta_data['strategic_potential'] * 500 + 20000
        )
        
        self.nta_data['monthly_profit'] = (
            self.nta_data['estimated_monthly_revenue'] * (1 - self.financial_params['variable_cost_rate'])
            - self.financial_params['fixed_cost_monthly']
        )
        
        self.nta_data['roi_months'] = (
            self.financial_params['initial_investment'] / 
            self.nta_data['monthly_profit'].clip(lower=1)
        )
        
        self.nta_data['safety_margin'] = (
            (self.nta_data['monthly_profit'] / 
             self.nta_data['estimated_monthly_revenue']) * 100
        )
        
        np.random.seed(47)
        self.nta_data['revenue_volatility'] = np.random.uniform(10, 30, len(self.nta_data))
        
        print(f"✓ 재무 분석 완료 (평균 ROI: {self.nta_data['roi_months'].mean():.1f}개월)")
    
    def classify_priorities(self):
        """우선순위 분류"""
        print("\n🎯 Step 9: 출점 우선순위 분류...")
        
        conditions = [
            (self.nta_data['strategic_potential'] >= 75) & (self.nta_data['roi_months'] <= 30),
            (self.nta_data['strategic_potential'] >= 75) & (self.nta_data['roi_months'] > 30),
            (self.nta_data['strategic_potential'] < 75) & (self.nta_data['roi_months'] <= 30),
            (self.nta_data['strategic_potential'] < 75) & (self.nta_data['roi_months'] > 30)
        ]
        
        priorities = ['Top Priority', 'High Potential', 'Quick ROI', 'Monitor']
        
        self.nta_data['priority_class'] = np.select(conditions, priorities, default='Monitor')
        
        self.nta_data['final_score'] = (
            self.nta_data['strategic_potential'] * 0.6 +
            (100 - (self.nta_data['roi_months'] / 60 * 100).clip(0, 100)) * 0.4
        )
        
        self.nta_data['rank'] = self.nta_data['final_score'].rank(ascending=False, method='min')
        
        print("✓ 우선순위 분류 완료")
        print(f"🏆 최우선 출점 대상: {len(self.nta_data[self.nta_data['priority_class']=='Top Priority'])}개 상권")
    
    def generate_visualizations(self):
        """시각화 생성"""
        print("\n📈 Step 10: 분석 결과 시각화...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 전략적 매트릭스
        ax1 = plt.subplot(2, 3, 1)
        colors = {'Top Priority': '#00ff00', 'High Potential': '#ffff00', 
                  'Quick ROI': '#ff9900', 'Monitor': '#ff0000'}
        
        for priority in colors.keys():
            mask = self.nta_data['priority_class'] == priority
            if mask.sum() > 0:
                ax1.scatter(self.nta_data.loc[mask, 'strategic_potential'],
                           self.nta_data.loc[mask, 'roi_months'],
                           c=colors[priority], label=priority, s=100, alpha=0.6)
        
        ax1.axvline(x=75, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Strategic Potential Score', fontsize=11)
        ax1.set_ylabel('ROI Period (Months)', fontsize=11)
        ax1.set_title('Strategic Decision Matrix', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. KSF 요소별 기여도
        ax2 = plt.subplot(2, 3, 2)
        top_areas = self.nta_data.nsmallest(5, 'rank')
        ksf_data = top_areas[['income_score', 'transit_score', 'competition_score', 
                               'mixed_use_score', 'subway_score']].T
        ksf_data.columns = top_areas['ntaname'].values
        
        ksf_data.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('KSF Factors', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Top 5 Areas - KSF Breakdown', fontsize=12, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.set_xticklabels(['Income', 'Transit', 'Competition', 'Mixed Use', 'Subway'], 
                            rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 재무 성과 예측
        ax3 = plt.subplot(2, 3, 3)
        top_10 = self.nta_data.nsmallest(10, 'rank')
        x_pos = np.arange(len(top_10))
        
        ax3_twin = ax3.twinx()
        bars = ax3.bar(x_pos, top_10['estimated_monthly_revenue'], color='skyblue', alpha=0.7, label='Revenue')
        line = ax3_twin.plot(x_pos, top_10['roi_months'], 'ro-', linewidth=2, markersize=8, label='ROI Months')
        
        ax3.set_xlabel('Top 10 Neighborhoods', fontsize=11)
        ax3.set_ylabel('Monthly Revenue ($)', fontsize=11, color='blue')
        ax3_twin.set_ylabel('ROI Period (Months)', fontsize=11, color='red')
        ax3.set_title('Financial Performance Forecast', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(range(1, 11), rotation=0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 4. 우선순위 분포
        ax4 = plt.subplot(2, 3, 4)
        priority_counts = self.nta_data['priority_class'].value_counts()
        wedges, texts, autotexts = ax4.pie(priority_counts, labels=priority_counts.index,
                                             autopct='%1.1f%%', colors=[colors[p] for p in priority_counts.index],
                                             startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Priority Distribution', fontsize=12, fontweight='bold')
        
        # 5. 리스크-수익 히트맵
        ax5 = plt.subplot(2, 3, 5)
        try:
            pivot_data = self.nta_data.pivot_table(
                values='final_score',
                index=pd.cut(self.nta_data['strategic_potential'], bins=5),
                columns=pd.cut(self.nta_data['roi_months'], bins=5),
                aggfunc='count'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Count'})
            ax5.set_title('Risk-Return Heatmap', fontsize=12, fontweight='bold')
            ax5.set_xlabel('ROI Period Range', fontsize=11)
            ax5.set_ylabel('Potential Score Range', fontsize=11)
        except:
            ax5.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
            ax5.set_title('Risk-Return Heatmap', fontsize=12, fontweight='bold')
        
        # 6. 상위 10개 상권 종합 점수
        ax6 = plt.subplot(2, 3, 6)
        top_10_sorted = top_10.sort_values('final_score', ascending=True)
        bars = ax6.barh(range(len(top_10_sorted)), top_10_sorted['final_score'], 
                        color=plt.cm.RdYlGn(top_10_sorted['final_score']/100))
        ax6.set_yticks(range(len(top_10_sorted)))
        ax6.set_yticklabels([f"{i+1}. {name[:20]}" for i, name in enumerate(top_10_sorted['ntaname'])],
                            fontsize=9)
        ax6.set_xlabel('Final Score', fontsize=11)
        ax6.set_title('Top 10 Neighborhoods - Final Ranking', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        for i, (idx, row) in enumerate(top_10_sorted.iterrows()):
            ax6.text(row['final_score'] + 1, i, f"{row['final_score']:.1f}", 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('nyc_franchise_analysis_report.png', dpi=300, bbox_inches='tight')
        print("✓ 시각화 저장 완료: nyc_franchise_analysis_report.png")
        plt.close()
    
    def generate_interactive_map(self):
        """실시간 인터랙티브 지도 생성"""
        print("\n🗺️ Step 11: 인터랙티브 지도 생성...")
        
        nyc_center = [40.7128, -74.0060]
        
        m = folium.Map(
            location=nyc_center,
            zoom_start=11,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        color_map = {
            'Top Priority': '#00ff00',
            'High Potential': '#ffff00',
            'Quick ROI': '#ff9900',
            'Monitor': '#ff0000'
        }
        
        for idx, row in self.nta_data.iterrows():
            centroid = row.geometry.centroid
            color = color_map.get(row['priority_class'], '#808080')
            
            popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h3 style="color: {color}; margin: 0;">#{int(row['rank'])} {row['ntaname']}</h3>
                <hr style="margin: 5px 0;">
                <b>🎯 Priority:</b> <span style="color: {color}; font-weight: bold;">{row['priority_class']}</span><br>
                <b>⭐ Final Score:</b> {row['final_score']:.1f}/100<br><br>
                
                <b style="color: #2E86AB;">📊 Strategic Potential: {row['strategic_potential']:.1f}/100</b><br>
                <div style="margin-left: 15px; font-size: 0.9em;">
                    • Income Score: {row['income_score']:.1f}<br>
                    • Transit Score: {row['transit_score']:.1f}<br>
                    • Competition Score: {row['competition_score']:.1f}<br>
                    • Mixed Use Score: {row['mixed_use_score']:.1f}<br>
                    • Subway Score: {row['subway_score']:.1f}
                </div><br>
                
                <b style="color: #A23B72;">💰 Financial Metrics</b><br>
                <div style="margin-left: 15px; font-size: 0.9em;">
                    • ROI Period: {row['roi_months']:.1f} months<br>
                    • Monthly Revenue: ${row['estimated_monthly_revenue']:,.0f}<br>
                    • Monthly Profit: ${row['monthly_profit']:,.0f}<br>
                    • Safety Margin: {row['safety_margin']:.1f}%
                </div><br>
                
                <b style="color: #F18F01;">📈 Market Data</b><br>
                <div style="margin-left: 15px; font-size: 0.9em;">
                    • Median Income: ${row['median_income']:,.0f}<br>
                    • Daily Pickups: {row['daily_pickups']:,.0f}<br>
                    • Competitors: {row['competitor_count']:.0f}<br>
                    • Subway Ridership: {row['subway_ridership']:,.0f}
                </div>
            </div>
            """
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=color, priority=row['priority_class']: {
                    'fillColor': color,
                    'color': color,
                    'weight': 3 if priority == 'Top Priority' else 2,
                    'fillOpacity': 0.4 if priority == 'Top Priority' else 0.2,
                    'opacity': 0.8
                },
                tooltip=f"{row['ntaname']} (Rank #{int(row['rank'])})"
            ).add_to(m)
            
            if row['rank'] <= 10:
                icon_size = 40 if row['rank'] <= 3 else 30
                
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    popup=folium.Popup(popup_html, max_width=350),
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-family: Arial;
                            font-size: 14px;
                            font-weight: bold;
                            color: white;
                            background-color: {color};
                            border: 3px solid white;
                            border-radius: 50%;
                            width: {icon_size}px;
                            height: {icon_size}px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            box-shadow: 0 0 10px rgba(0,0,0,0.5);
                        ">
                            {int(row['rank'])}
                        </div>
                    """)
                ).add_to(m)
        
        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            width: 220px;
            background-color: white;
            border: 2px solid grey;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial;
            font-size: 12px;
            z-index: 9999;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        ">
            <h4 style="margin: 0 0 10px 0; text-align: center;">Priority Legend</h4>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: #00ff00; margin-right: 10px;"></div>
                <span>Top Priority</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: #ffff00; margin-right: 10px;"></div>
                <span>High Potential</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: #ff9900; margin-right: 10px;"></div>
                <span>Quick ROI</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: #ff0000; margin-right: 10px;"></div>
                <span>Monitor</span>
            </div>
            <hr style="margin: 10px 0;">
            <div style="text-align: center; font-size: 11px; color: #666;">
                Numbers indicate rank<br>
                Click areas for details
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        plugins.MiniMap(toggle_display=True, position='topleft').add_to(m)
        plugins.Fullscreen(position='topleft', force_separate_button=True).add_to(m)
        plugins.MeasureControl(position='topleft', primary_length_unit='miles').add_to(m)
        
        map_filename = 'nyc_franchise_interactive_map.html'
        m.save(map_filename)
        
        print(f"✓ 인터랙티브 지도 생성 완료: {map_filename}")
        print(f"  - 총 {len(self.nta_data)}개 상권 표시")
        print(f"  - Top 10 상권에 순위 마커 표시")
        
        return m
    
    def generate_priority_map(self):
        """우선순위별 별도 지도 생성"""
        print("\n🎯 Step 12: 우선순위 상세 지도 생성...")
        
        top_priority = self.nta_data[self.nta_data['priority_class'] == 'Top Priority']
        
        if len(top_priority) == 0:
            print("⚠ Top Priority 상권이 없습니다.")
            return None
        
        nyc_center = [40.7128, -74.0060]
        m = folium.Map(location=nyc_center, zoom_start=12, tiles='CartoDB positron')
        
        heat_data = []
        
        for idx, row in top_priority.iterrows():
            centroid = row.geometry.centroid
            
            popup_html = f"""
            <div style="font-family: Arial; width: 350px;">
                <h2 style="color: #00ff00; margin: 0;">🏆 #{int(row['rank'])} {row['ntaname']}</h2>
                <h3 style="color: #00ff00; margin: 5px 0;">TOP PRIORITY LOCATION</h3>
                <hr>
                <b>Final Score:</b> {row['final_score']:.1f}/100<br>
                <b>Strategic Potential:</b> {row['strategic_potential']:.1f}/100<br>
                <b>ROI Period:</b> {row['roi_months']:.1f} months<br>
                <b>Monthly Revenue:</b> ${row['estimated_monthly_revenue']:,.0f}<br>
                <b>Monthly Profit:</b> ${row['monthly_profit']:,.0f}<br>
                <b>Safety Margin:</b> {row['safety_margin']:.1f}%
            </div>
            """
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    'fillColor': '#00ff00',
                    'color': '#00ff00',
                    'weight': 4,
                    'fillOpacity': 0.5,
                    'opacity': 1.0
                }
            ).add_to(m)
            
            folium.Marker(
                location=[centroid.y, centroid.x],
                popup=folium.Popup(popup_html, max_width=400),
                icon=folium.Icon(color='green', icon='star', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-family: Arial Black;
                        font-size: 20px;
                        color: white;
                        background-color: #00aa00;
                        border: 4px solid white;
                        border-radius: 50%;
                        width: 50px;
                        height: 50px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        box-shadow: 0 0 15px rgba(0,255,0,0.8);
                    ">
                        {int(row['rank'])}
                    </div>
                """)
            ).add_to(m)
            
            heat_data.append([centroid.y, centroid.x, row['final_score']])
        
        plugins.HeatMap(
            heat_data,
            radius=25,
            blur=35,
            max_zoom=13,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        title_html = """
        <div style="
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background-color: white;
            padding: 15px 30px;
            border: 3px solid #00ff00;
            border-radius: 10px;
            font-family: Arial Black;
            font-size: 18px;
            z-index: 9999;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);
        ">
            🏆 TOP PRIORITY LOCATIONS - NYC FRANCHISE EXPANSION
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        stats_html = f"""
        <div style="
            position: fixed;
            top: 70px;
            right: 50px;
            background-color: white;
            padding: 15px;
            border: 2px solid #00ff00;
            border-radius: 5px;
            font-family: Arial;
            font-size: 12px;
            z-index: 9999;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        ">
            <h4 style="margin: 0 0 10px 0; color: #00aa00;">PORTFOLIO SUMMARY</h4>
            <b>Total Locations:</b> {len(top_priority)}<br>
            <b>Avg ROI:</b> {top_priority['roi_months'].mean():.1f} months<br>
            <b>Avg Potential:</b> {top_priority['strategic_potential'].mean():.1f}/100<br>
            <b>Total Investment:</b> ${len(top_priority) * 150000:,.0f}<br>
            <b>Monthly Profit:</b> ${top_priority['monthly_profit'].sum():,.0f}<br>
            <hr style="margin: 10px 0;">
            <b style="color: green;">Year 1 Revenue:</b><br>
            <b style="font-size: 14px;">${top_priority['monthly_profit'].sum() * 12:,.0f}</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(stats_html))
        
        priority_map_filename = 'nyc_franchise_top_priority_map.html'
        m.save(priority_map_filename)
        
        print(f"✓ 우선순위 지도 생성 완료: {priority_map_filename}")
        print(f"  - {len(top_priority)}개 Top Priority 상권 상세 표시")
        
        return m
    
    def generate_executive_summary(self):
        """경영진 요약 보고서 생성"""
        print("\n" + "="*80)
        print("📋 EXECUTIVE SUMMARY - NYC FRANCHISE EXPANSION STRATEGY")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Total Neighborhoods Analyzed: {len(self.nta_data)}")
        print("\n" + "-"*80)
        
        top_priority = self.nta_data[self.nta_data['priority_class'] == 'Top Priority']
        print(f"\n🎯 TOP PRIORITY LOCATIONS: {len(top_priority)} neighborhoods")
        print("-"*80)
        
        top_5 = self.nta_data.nsmallest(5, 'rank')
        for idx, row in top_5.iterrows():
            print(f"\n#{int(row['rank'])} {row['ntaname']}")
            print(f"   Strategic Potential: {row['strategic_potential']:.1f}/100")
            print(f"   ROI Period: {row['roi_months']:.1f} months")
            print(f"   Est. Monthly Revenue: ${row['estimated_monthly_revenue']:,.0f}")
            print(f"   Est. Monthly Profit: ${row['monthly_profit']:,.0f}")
            print(f"   Safety Margin: {row['safety_margin']:.1f}%")
            print(f"   Priority: {row['priority_class']}")
        
        print("\n" + "-"*80)
        print("💰 FINANCIAL SUMMARY (Top Priority Locations)")
        print("-"*80)
        if len(top_priority) > 0:
            print(f"Average ROI Period: {top_priority['roi_months'].mean():.1f} months")
            print(f"Average Monthly Revenue: ${top_priority['estimated_monthly_revenue'].mean():,.0f}")
            print(f"Average Monthly Profit: ${top_priority['monthly_profit'].mean():,.0f}")
            print(f"Average Safety Margin: {top_priority['safety_margin'].mean():.1f}%")
        
        print("\n" + "-"*80)
        print("⚠️ RISK ASSESSMENT")
        print("-"*80)
        print(f"Low Risk (<25 months ROI): {len(self.nta_data[self.nta_data['roi_months'] < 25])} areas")
        print(f"Medium Risk (25-35 months): {len(self.nta_data[(self.nta_data['roi_months'] >= 25) & (self.nta_data['roi_months'] <= 35)])} areas")
        print(f"High Risk (>35 months): {len(self.nta_data[self.nta_data['roi_months'] > 35])} areas")
        
        print("\n" + "-"*80)
        print("📌 STRATEGIC RECOMMENDATIONS")
        print("-"*80)
        print("1. Immediate Action: Focus on Top 3 priority locations for Q1 expansion")
        print("2. Portfolio Approach: Mix 2-3 'Top Priority' with 1-2 'Quick ROI' locations")
        print("3. Risk Mitigation: Maintain safety margin above 15% for all new locations")
        print("4. Market Entry: Prioritize areas with strategic potential >80 points")
        print("5. Competitive Strategy: Target neighborhoods with competition score >70")
        
        print("\n" + "="*80 + "\n")
    
    def export_detailed_report(self):
        """상세 분석 결과 CSV 내보내기"""
        output_columns = [
            'rank', 'ntaname', 'priority_class', 'final_score',
            'strategic_potential', 'roi_months', 'estimated_monthly_revenue',
            'monthly_profit', 'safety_margin', 'revenue_volatility',
            'income_score', 'transit_score', 'competition_score',
            'mixed_use_score', 'subway_score',
            'median_income', 'daily_pickups', 'competitor_count',
            'commercial_ratio', 'subway_ridership'
        ]
        
        export_df = self.nta_data[output_columns].sort_values('rank')
        export_df.to_csv('nyc_franchise_detailed_analysis.csv', index=False, encoding='utf-8-sig')
        print("✓ 상세 분석 보고서 저장: nyc_franchise_detailed_analysis.csv")
    
    def run_complete_analysis(self):
        """전체 분석 프로세스 실행"""
        print("\n" + "="*80)
        print("🚀 NYC FRANCHISE LOCATION ANALYSIS SYSTEM")
        print("="*80 + "\n")
        
        self.fetch_nta_boundaries()
        self.calculate_income_score()
        self.calculate_transit_density()
        self.calculate_competition_index()
        self.calculate_mixed_use()
        self.calculate_subway_access()
        
        self.calculate_strategic_potential()
        self.calculate_financial_risk()
        self.classify_priorities()
        
        self.generate_visualizations()
        self.generate_interactive_map()
        self.generate_priority_map()
        self.generate_executive_summary()
        self.export_detailed_report()
        
        print("\n✅ 분석 완료! 모든 결과물이 생성되었습니다.")
        print("   - nyc_franchise_analysis_report.png (시각화)")
        print("   - nyc_franchise_detailed_analysis.csv (상세 데이터)")
        print("   - nyc_franchise_interactive_map.html (전체 인터랙티브 지도)")
        print("   - nyc_franchise_top_priority_map.html (우선순위 지도)")

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    analyzer = NYCFranchiseAnalyzer()
    analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("📊 DETAILED ANALYSIS - Top Recommendation")
    print("="*80)
    
    top_location = analyzer.nta_data.nsmallest(1, 'rank').iloc[0]
    
    print(f"\n🏆 #1 추천 상권: {top_location['ntaname']}")
    print("\n[전략적 강점]")
    print(f"  • 소득 수준: ${top_location['median_income']:,.0f} (점수: {top_location['income_score']:.1f})")
    print(f"  • 교통 밀도: 일일 {top_location['daily_pickups']:,.0f}건 (점수: {top_location['transit_score']:.1f})")
    print(f"  • 경쟁 환경: {top_location['competitor_count']:.0f}개 경쟁사 (점수: {top_location['competition_score']:.1f})")
    print(f"  • 토지 이용: 상업 {top_location['commercial_ratio']:.1%} (점수: {top_location['mixed_use_score']:.1f})")
    print(f"  • 지하철: 일일 {top_location['subway_ridership']:,.0f}명 (점수: {top_location['subway_score']:.1f})")
    
    print("\n[재무 전망]")
    print(f"  • 예상 월 매출: ${top_location['estimated_monthly_revenue']:,.0f}")
    print(f"  • 예상 월 순이익: ${top_location['monthly_profit']:,.0f}")
    print(f"  • ROI 회수 기간: {top_location['roi_months']:.1f}개월")
    print(f"  • 안전 마진: {top_location['safety_margin']:.1f}%")
    print(f"  • 매출 변동성: {top_location['revenue_volatility']:.1f}%")
    
    print("\n[5개년 누적 수익 예측]")
    monthly_profit = top_location['monthly_profit']
    initial_investment = analyzer.financial_params['initial_investment']
    
    for year in range(1, 6):
        months = year * 12
        cumulative_profit = monthly_profit * months - initial_investment
        roi_pct = (cumulative_profit / initial_investment) * 100
        print(f"  Year {year}: ${cumulative_profit:,.0f} (ROI: {roi_pct:+.1f}%)")
    
    print("\n" + "="*80)


    # 시나리오 분석
    print("\n📉 SCENARIO ANALYSIS - Sensitivity Testing")
    print("="*80)
    
    scenarios = {
        'Best Case (+20% Revenue)': 1.2,
        'Base Case': 1.0,
        'Worst Case (-20% Revenue)': 0.8
    }
    
    print(f"\nLocation: {top_location['ntaname']}")
    print("-"*80)
    
    for scenario_name, multiplier in scenarios.items():
        adjusted_revenue = top_location['estimated_monthly_revenue'] * multiplier
        adjusted_profit = (adjusted_revenue * (1 - analyzer.financial_params['variable_cost_rate'])
                          - analyzer.financial_params['fixed_cost_monthly'])
        adjusted_roi = initial_investment / max(adjusted_profit, 1)
        
        print(f"\n{scenario_name}:")
        print(f"  Monthly Revenue: ${adjusted_revenue:,.0f}")
        print(f"  Monthly Profit: ${adjusted_profit:,.0f}")
        print(f"  ROI Period: {adjusted_roi:.1f} months")
        print(f"  Break-even: {'✓ PASS' if adjusted_roi <= 36 else '✗ FAIL'}")
    
    print("\n" + "="*80)
    
    # 포트폴리오 추천
    print("\n🎯 RECOMMENDED EXPANSION PORTFOLIO")
    print("="*80)
    
    print("\n[Phase 1 - Immediate (Q1 2026)]")
    phase1 = analyzer.nta_data[
        (analyzer.nta_data['priority_class'] == 'Top Priority') &
        (analyzer.nta_data['rank'] <= 3)
    ].sort_values('rank')
    
    total_investment_p1 = len(phase1) * initial_investment
    total_monthly_profit_p1 = phase1['monthly_profit'].sum()
    
    for idx, row in phase1.iterrows():
        print(f"  {int(row['rank'])}. {row['ntaname']}")
        print(f"     - Investment: ${initial_investment:,.0f}")
        print(f"     - Monthly Profit: ${row['monthly_profit']:,.0f}")
        print(f"     - ROI: {row['roi_months']:.1f} months")
    
    print(f"\n  Phase 1 Total Investment: ${total_investment_p1:,.0f}")
    print(f"  Phase 1 Monthly Profit: ${total_monthly_profit_p1:,.0f}")
    print(f"  Phase 1 Avg ROI: {phase1['roi_months'].mean():.1f} months")
    
    print("\n[Phase 2 - Secondary (Q2-Q3 2026)]")
    phase2 = analyzer.nta_data[
        (analyzer.nta_data['priority_class'] == 'Top Priority') &
        (analyzer.nta_data['rank'] > 3) &
        (analyzer.nta_data['rank'] <= 6)
    ].sort_values('rank')
    
    for idx, row in phase2.iterrows():
        print(f"  {int(row['rank'])}. {row['ntaname']}")
        print(f"     - Strategic Potential: {row['strategic_potential']:.1f}")
        print(f"     - ROI: {row['roi_months']:.1f} months")
    
    print("\n[Phase 3 - Quick ROI Opportunities (Q4 2026)]")
    phase3 = analyzer.nta_data[
        (analyzer.nta_data['priority_class'] == 'Quick ROI') &
        (analyzer.nta_data['roi_months'] <= 24)
    ].nsmallest(3, 'roi_months')
    
    for idx, row in phase3.iterrows():
        print(f"  {int(row['rank'])}. {row['ntaname']}")
        print(f"     - Fast ROI: {row['roi_months']:.1f} months")
        print(f"     - Monthly Profit: ${row['monthly_profit']:,.0f}")
    
    # 전체 포트폴리오 요약
    total_locations = len(phase1) + len(phase2) + len(phase3)
    total_investment = total_locations * initial_investment
    total_monthly = phase1['monthly_profit'].sum() + phase2['monthly_profit'].sum() + phase3['monthly_profit'].sum()
    
    print("\n" + "-"*80)
    print("PORTFOLIO SUMMARY:")
    print(f"  Total Locations: {total_locations}")
    print(f"  Total Investment: ${total_investment:,.0f}")
    print(f"  Combined Monthly Profit: ${total_monthly:,.0f}")
    print(f"  Portfolio ROI: {(total_investment / total_monthly):.1f} months")
    print(f"  Annual Profit (Year 1): ${total_monthly * 12:,.0f}")
    
    print("\n" + "="*80)
    
    # 실행 계획
    print("\n📅 IMPLEMENTATION ROADMAP")
    print("="*80)
    
    print("""
Q1 2026 (Jan-Mar):
  ✓ Phase 1 locations site selection and lease negotiation
  ✓ Financial due diligence and final approval
  ✓ Begin construction/renovation (Top 3 locations)
  
Q2 2026 (Apr-Jun):
  ✓ Phase 1 grand openings (3 locations)
  ✓ Phase 2 location scouting and evaluation
  ✓ Monitor Phase 1 performance metrics
  
Q3 2026 (Jul-Sep):
  ✓ Phase 2 development begins (3 locations)
  ✓ Phase 1 performance review and optimization
  ✓ Phase 3 opportunity identification
  
Q4 2026 (Oct-Dec):
  ✓ Phase 2 grand openings
  ✓ Phase 3 Quick ROI locations launch
  ✓ Full portfolio performance analysis
  
Q1 2027:
  ✓ Portfolio-wide optimization
  ✓ Expansion planning based on Year 1 results
  ✓ Additional market penetration strategy
""")
    
    print("="*80)
    
    # KPI 추적 프레임워크
    print("\n📊 KEY PERFORMANCE INDICATORS (KPIs) TRACKING")
    print("="*80)
    
    print("""
FINANCIAL KPIs:
  • Monthly Revenue vs. Forecast (Target: ±10%)
  • Gross Profit Margin (Target: >65%)
  • Net Profit Margin (Target: >20%)
  • ROI Achievement Rate (Target: Within 30 months)
  • Cash Flow Positive Date (Target: Month 3)
  
OPERATIONAL KPIs:
  • Daily Transaction Count
  • Average Transaction Value
  • Customer Acquisition Cost
  • Customer Retention Rate (Target: >60%)
  • Staff Productivity Metrics
  
MARKET KPIs:
  • Market Share in Neighborhood
  • Brand Awareness Score
  • Customer Satisfaction (Target: >4.5/5.0)
  • Competitor Response Analysis
  • Local Partnership Development
  
RISK INDICATORS:
  • Revenue Volatility (Target: <20%)
  • Fixed Cost Coverage Ratio (Target: >150%)
  • Lease Obligation vs. Revenue
  • Supply Chain Disruption Impact
  • Regulatory Compliance Status
""")
    
    print("="*80)
    
    # 최종 권고사항
    print("\n🎯 FINAL RECOMMENDATIONS & ACTION ITEMS")
    print("="*80)
    
    print("""
IMMEDIATE ACTIONS (Next 30 Days):
  1. Approve Phase 1 budget: ${:,.0f}
  2. Initiate site visits for Top 3 locations
  3. Engage real estate broker for lease negotiations
  4. Begin architectural planning and permitting process
  5. Establish local vendor partnerships
  
STRATEGIC PRIORITIES:
  1. Focus on neighborhoods with >75 strategic potential
  2. Maintain portfolio ROI target of <30 months
  3. Diversify across Manhattan submarkets
  4. Balance high-potential with quick-ROI locations
  5. Build scalable operational infrastructure
  
RISK MITIGATION:
  1. Secure flexible lease terms (5yr + 5yr option)
  2. Implement comprehensive insurance coverage
  3. Establish contingency fund (20% of investment)
  4. Develop exit strategy for underperforming locations
  5. Monitor market conditions monthly
  
SUCCESS FACTORS:
  1. Strong local market knowledge
  2. Efficient supply chain management
  3. Excellent customer service standards
  4. Aggressive but sustainable growth pace
  5. Data-driven decision making
""".format(total_investment_p1))
    
    print("="*80)
    print("\n✅ ANALYSIS COMPLETE - All deliverables generated successfully!")
    print("\nGenerated Files:")
    print("  1. nyc_franchise_analysis_report.png - Executive dashboard")
    print("  2. nyc_franchise_detailed_analysis.csv - Complete data export")
    print("\n💼 Ready for strategic planning meeting.\n")
    print("="*80)