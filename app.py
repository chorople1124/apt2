import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="서울 아파트 월세 예측 (Random Forest)", layout="wide")

st.title("🏙️ 서울 아파트 월세 예측 · 건축년도별 평균 월세 시각화")
st.write("랜덤 포레스트 회귀 모델을 이용해 서울 아파트 월세를 예측하고, **건축년도(시대)별 평균 월세 추세**를 그래프로 보여주는 앱입니다.")

# --------------------------------------------------------------------
# 1. CSV 불러오기: 업로드 또는 기본 파일
# --------------------------------------------------------------------
st.sidebar.header("📁 데이터 불러오기")

uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (.csv)", type=["csv"])

# 👉 기본 CSV 파일 경로 (원하면 고쳐 쓰기)
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
# 2. 컬럼 이름 고정 (질문에서 준 컬럼명 그대로 사용)
# --------------------------------------------------------------------
COL_SIGUNGU = "시군구"
COL_AREA = "평수"
COL_RENT = "월세금(만원)"
COL_YEAR_BUILT = "건축년도"

required_cols = [COL_SIGUNGU, COL_AREA, COL_RENT, COL_YEAR_BUILT]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"다음 컬럼을 찾을 수 없습니다: {missing_cols}\nCSV 헤더 이름이 정확히 일치하는지 확인해주세요.")
    st.stop()

# 타입 정리: 숫자형 컬럼들을 숫자로 캐스팅
for col in [COL_AREA, COL_RENT, COL_YEAR_BUILT]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 건축년도는 정수형으로 보는 게 자연스러우니 소수점 제거
df[COL_YEAR_BUILT] = df[COL_YEAR_BUILT].round(0).astype("Int64")

# --------------------------------------------------------------------
# 3. 타깃/피처 설정
# --------------------------------------------------------------------
TARGET_COL = COL_RENT
numeric_features = [COL_AREA, COL_YEAR_BUILT]
categorical_features = [COL_SIGUNGU]

# 필요한 컬럼만 복사
df_model = df[numeric_features + categorical_features + [TARGET_COL]].copy()

# 타깃 결측치 제거
df_model = df_model.dropna(subset=[TARGET_COL])

# 숫자형 결측치는 중간값으로 채우기
for col in numeric_features:
    if df_model[col].isna().sum() > 0:
        df_model[col] = df_model[col].fillna(df_model[col].median())

# 범주형 결측치는 '미상' 같은 값으로 채우기
for col in categorical_features:
    if df_model[col].isna().sum() > 0:
        df_model[col] = df_model[col].fillna("미상")

# --------------------------------------------------------------------
# 4. 원-핫 인코딩
# --------------------------------------------------------------------
df_model_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

X = df_model_encoded.drop(columns=[TARGET_COL])
y = df_model_encoded[TARGET_COL]

# 건축년도는 그래프용으로 따로 보관
years_built = df_model[COL_YEAR_BUILT].values

# --------------------------------------------------------------------
# 5. 사이드바에서 모델 설정
# --------------------------------------------------------------------
st.sidebar.header("🧠 Random Forest 설정")
n_estimators = st.sidebar.slider("트리 개수 (n_estimators)", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("최대 깊이 (max_depth, 0 = 제한 없음)", 0, 30, 0, step=1)
max_depth_param = None if max_depth == 0 else max_depth
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, step=0.05)

# --------------------------------------------------------------------
# 6. 학습/검증 데이터 분리 후 학습
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth_param,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------------------------
# 7. 성능 평가
# --------------------------------------------------------------------
y_pred_test = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

st.subheader("🧪 모델 성능 (테스트 데이터 기준)")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE (월세 예측 오차, 만원 단위)", f"{rmse:,.2f} 만원")
with col2:
    st.write(f"테스트 데이터 개수: {len(y_test)}")

# --------------------------------------------------------------------
# 8. 전체 데이터에 대해 예측 + 건축년도별 평균 계산
# --------------------------------------------------------------------
all_pred = model.predict(X)

df_result = pd.DataFrame({
    COL_YEAR_BUILT: years_built,
    "실제_월세(만원)": y,
    "예측_월세(만원)": all_pred
})

# 건축년도별 평균
yearly_actual_mean = df_result.groupby(COL_YEAR_BUILT)["실제_월세(만원)"].mean()
yearly_pred_mean = df_result.groupby(COL_YEAR_BUILT)["예측_월세(만원)"].mean()

st.subheader("📊 건축년도별 평균 월세 (실제 vs 예측)")
summary_df = pd.DataFrame({
    "실제 평균 월세(만원)": yearly_actual_mean.round(2),
    "예측 평균 월세(만원)": yearly_pred_mean.round(2)
})
st.dataframe(summary_df)

# --------------------------------------------------------------------
# 9. 그래프: 건축년도(시대)별 평균 월세 추세
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

years_sorted = sorted(yearly_actual_mean.index.dropna())

ax.plot(
    years_sorted,
    [yearly_actual_mean[y] for y in years_sorted],
    marker="o",
    label="실제 평균 월세(만원)"
)
ax.plot(
    years_sorted,
    [yearly_pred_mean[y] for y in years_sorted],
    marker="s",
    linestyle="--",
    label="예측 평균 월세(만원)"
)

ax.set_title("건축년도별 평균 아파트 월세 (실제 vs 랜덤 포레스트 예측)")
ax.set_xlabel("건축년도")
ax.set_ylabel("월세 (만원)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.caption("※ 월세 단위는 CSV에서 준 그대로(만원 기준) 사용했습니다. 데이터 범위와 품질에 따라 그래프 모양과 예측 성능이 달라질 수 있습니다.")
