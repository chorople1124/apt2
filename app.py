import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="서울 아파트 월세 예측 (Random Forest)", layout="wide")

st.title("🏙️ 서울 아파트 월세 예측 · 시대별 평균 월세 시각화")
st.write("랜덤 포레스트 회귀 모델을 이용해 서울 아파트 월세를 예측하고, 연도별 평균 월세 추세를 그래프로 보여주는 앱입니다.")

# --------------------------------------------------------------------
# 1. CSV 불러오기: 업로드 또는 기본 파일
# --------------------------------------------------------------------
st.sidebar.header("📁 데이터 불러오기")

uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (.csv)", type=["csv"])

# 👉 기본 CSV 파일 경로 (원하시면 수정)
default_path = "csv.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("업로드한 CSV 파일을 사용합니다.")
else:
    try:
        df = pd.read_csv(default_path)
        st.sidebar.info(f"업로드된 파일이 없어, 기본 파일 `{default_path}` 를 사용합니다.")
    except FileNotFoundError:
        st.error("CSV 파일이 없습니다. 사이드바에서 CSV를 업로드하거나, 코드의 `default_path` 경로를 수정해주세요.")
        st.stop()

st.subheader("원본 데이터 미리보기")
st.dataframe(df.head())

# --------------------------------------------------------------------
# 2. 컬럼 이름 설정 (여기만 본인 CSV에 맞게 수정)
# --------------------------------------------------------------------
# ⚠️ 아래 네 줄은 꼭 본인 CSV 컬럼명에 맞게 바꿔주세요.
CONTRACT_DATE_COL = "건축년도"   # 계약일 컬럼명 (예: '계약일', 'date' 등)
TARGET_COL        = "월세금(만원)    # 월세 컬럼명   (예: '월세', 'rent' 등)

# 예시로 사용하는 피처들 - 실제 CSV에 존재하는 것만 자동으로 사용
CANDIDATE_NUMERIC_FEATURES = [
    "deposit",      # 보증금
    "area",         # 전용면적(m²)
    "floor",        # 층
    "built_year"    # 준공연도
]

CANDIDATE_CATEGORICAL_FEATURES = [
    "gu",           # 자치구
    "dong"          # 동
]

# --------------------------------------------------------------------
# 3. 날짜 처리: 연도/월 추출
# --------------------------------------------------------------------
if CONTRACT_DATE_COL not in df.columns:
    st.error(f"'{CONTRACT_DATE_COL}' 컬럼을 찾을 수 없습니다. 코드 상단의 CONTRACT_DATE_COL 이름을 CSV에 맞게 수정해주세요.")
    st.stop()

df[CONTRACT_DATE_COL] = pd.to_datetime(df[CONTRACT_DATE_COL], errors="coerce")
if df[CONTRACT_DATE_COL].isna().all():
    st.error(f"'{CONTRACT_DATE_COL}' 컬럼을 날짜로 변환할 수 없습니다. 날짜 형식을 확인해주세요.")
    st.stop()

df["year"] = df[CONTRACT_DATE_COL].dt.year
df["month"] = df[CONTRACT_DATE_COL].dt.month

# --------------------------------------------------------------------
# 4. 타깃(월세)과 사용 가능한 피처 자동 선택
# --------------------------------------------------------------------
if TARGET_COL not in df.columns:
    st.error(f"'{TARGET_COL}' 컬럼을 찾을 수 없습니다. 코드 상단의 TARGET_COL 이름을 CSV에 맞게 수정해주세요.")
    st.stop()

# 실제 있는 컬럼만 사용
numeric_features = [c for c in CANDIDATE_NUMERIC_FEATURES if c in df.columns]
categorical_features = [c for c in CANDIDATE_CATEGORICAL_FEATURES if c in df.columns]

st.sidebar.header("⚙️ 사용 피처 설정")
st.sidebar.write("※ 아래는 CSV에 실제 존재하는 컬럼만 표시됩니다.")

sel_numeric = st.sidebar.multiselect(
    "숫자형 피처 선택",
    options=numeric_features,
    default=numeric_features
)

sel_categorical = st.sidebar.multiselect(
    "범주형 피처 선택 (원-핫 인코딩)",
    options=categorical_features,
    default=categorical_features
)

if len(sel_numeric) + len(sel_categorical) == 0:
    st.error("최소 1개 이상의 피처를 선택해야 합니다.")
    st.stop()

# --------------------------------------------------------------------
# 5. 결측치 처리 및 원-핫 인코딩
# --------------------------------------------------------------------
df_model = df[sel_numeric + sel_categorical + [TARGET_COL, "year"]].copy()

# 타깃 결측치 제거
df_model = df_model.dropna(subset=[TARGET_COL])

# 숫자형 결측치는 중간값으로 채우기
for col in sel_numeric:
    if df_model[col].isna().sum() > 0:
        df_model[col] = df_model[col].fillna(df_model[col].median())

# 범주형 원-핫 인코딩
if sel_categorical:
    df_model = pd.get_dummies(df_model, columns=sel_categorical, drop_first=True)

# --------------------------------------------------------------------
# 6. 학습/검증 데이터 분리
# --------------------------------------------------------------------
X = df_model.drop(columns=[TARGET_COL, "year"])
y = df_model[TARGET_COL]
years = df_model["year"]

# 학습 설정
st.sidebar.header("🧠 Random Forest 설정")
n_estimators = st.sidebar.slider("트리 개수 (n_estimators)", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("최대 깊이 (max_depth, 0 = 제한 없음)", 0, 30, 0, step=1)
max_depth_param = None if max_depth == 0 else max_depth
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, step=0.05)

X_train, X_test, y_train, y_test, year_train, year_test = train_test_split(
    X, y, years, test_size=test_size, random_state=42
)

# --------------------------------------------------------------------
# 7. 랜덤 포레스트 학습
# --------------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth_param,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 테스트 셋 성능
y_pred_test = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

st.subheader("🧪 모델 성능 (테스트 데이터)")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE (월세 예측 오차)", f"{rmse:,.0f} 원")
with col2:
    st.write(f"테스트 데이터 개수: {len(y_test)}")

# --------------------------------------------------------------------
# 8. 전체 데이터 예측 및 연도별 평균 계산
# --------------------------------------------------------------------
df_model["pred_rent"] = model.predict(X)

yearly_actual_mean = df_model.groupby("year")[TARGET_COL].mean()
yearly_pred_mean = df_model.groupby("year")["pred_rent"].mean()

st.subheader("📈 연도별 평균 월세 (실제 vs 예측)")
st.write("연도별로 실제 평균 월세와 랜덤 포레스트 예측 평균 월세를 비교합니다.")

# 표로 먼저 보여주기
summary_df = pd.DataFrame({
    "실제 평균 월세": yearly_actual_mean.round(0),
    "예측 평균 월세": yearly_pred_mean.round(0)
})
st.dataframe(summary_df)

# --------------------------------------------------------------------
# 9. 그래프 그리기 (시대별 평균 아파트 월세값)
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

years_sorted = sorted(yearly_actual_mean.index)

ax.plot(
    years_sorted,
    [yearly_actual_mean[yr] for yr in years_sorted],
    marker="o",
    label="실제 평균 월세"
)
ax.plot(
    years_sorted,
    [yearly_pred_mean[yr] for yr in years_sorted],
    marker="s",
    linestyle="--",
    label="예측 평균 월세 (Random Forest)"
)

ax.set_title("서울 아파트 연도별 평균 월세 (실제 vs 예측)")
ax.set_xlabel("연도")
ax.set_ylabel("월세 (원)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.caption("※ 데이터 품질(계약년도 범위, 이상값, 샘플 수 등)에 따라 그래프 모양과 예측 성능이 달라질 수 있습니다.")
