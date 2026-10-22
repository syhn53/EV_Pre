# %%
import pandas as pd

# %%
from pathlib import Path
import pandas as pd

file_path = Path(
    r"C:\test\01_원본데이터\서울시 소유 전기차충전기 일별 시간별 충전현황(2025년).xlsx"
)

print("파일 존재 여부:", file_path.exists())

# %%
df = pd.read_excel(file_path, header=3)

# %%
# 각 열의 고유값 개수 확인 충전소별 데이터 개수 확인  
for col in df.columns:
    print(f"{col}: {df[col].nunique()}개")

# %%
# 충전소별 데이터 개수
print(df["충전소명"].value_counts())

# %%
# 중복을 제외한 충전소명 목록
print(df["충전소명"].unique())

# 충전소 개수
print("충전소 개수:", df["충전소명"].nunique())

# %%
# 충전소명별로 데이터 그룹화
groups = df.groupby("충전소명")

# 충전소 개수 확인
print("충전소 개수:", df["충전소명"].nunique())

# %%
# 첫 번째 충전소 이름 확인
name = df["충전소명"].unique()[0]
print(name)

# 해당 충전소 데이터만 보기
display(df[df["충전소명"] == name])

# %%
for name, data in df.groupby("충전소명"):
    data.to_excel(f"{name}.xlsx", index=False)

# %%
import pandas as pd

# 시작 시간과 종료 시간을 날짜/시간 형식으로 변환
df["충전시작시간"] = pd.to_datetime(df["충전시작시간"])
df["충전종료시간"] = pd.to_datetime(df["충전종료시간"])

# 종료 시간 - 시작 시간 → 분 단위로 변환
df["충전시간_분"] = (
    df["충전종료시간"] - df["충전시작시간"]
).dt.total_seconds() / 60

# 결과 확인
display(df.head())

# %%
# 행과 열 개수 확인
print(df.shape)

# %%
print(df.columns.tolist())

# %%
# 충전량을 숫자형으로 변환
df["충전량(kW)"] = pd.to_numeric(
    df["충전량(kW)"], errors="coerce"
)

# 최솟값과 최댓값 확인
print("충전량 최솟값:", df["충전량(kW)"].min())
print("충전량 최댓값:", df["충전량(kW)"].max())

# %%
# 충전량이 0보다 작은 데이터
negative_df = df[df["충전량(kW)"] < 0]

print("음수 데이터 개수:", len(negative_df))
display(negative_df.head(10))

# %%
# 충전량이 절댓값 기준으로 100보다 큰 데이터 확인
display(
    df[df["충전량(kW)"].abs() > 100]
    .sort_values("충전량(kW)")
)

# %%
# 음수 충전량 제외한 분석용 데이터
df_clean = df[df["충전량(kW)"] >= 0].copy()

print("원본 데이터:", len(df))
print("분석용 데이터:", len(df_clean))
print("제외한 데이터:", len(df) - len(df_clean))

# %%
# 분석용 데이터의 충전량 분포 확인
print(df_clean["충전량(kW)"].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
))

# 충전량이 큰 데이터 확인
display(
    df_clean.nlargest(10, "충전량(kW)")[
        ["충전소명", "충전시작시간", "충전종료시간",
         "충전량(kW)", "충전시간_분"]
    ]
)

# %%
display(
    df_clean[df_clean["충전량(kW)"] > 1000][
        ["충전소명", "충전시작시간", "충전종료시간",
         "충전량(kW)", "충전시간_분"]
    ].sort_values("충전량(kW)", ascending=False)
)

# %%
# 원본 데이터에서 음수와 극단값만 제외
df_model = df[
    (df["충전량(kW)"] >= 0) &
    (df["충전량(kW)"] <= 1000)
].copy()

print("원본 데이터:", len(df))
print("분석용 데이터:", len(df_model))
print("제외한 데이터:", len(df) - len(df_model))

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []

for name, data in df_model.groupby("충전소명"):

    data = data[["충전시간_분", "충전량(kW)"]].dropna()

    # 데이터가 2개 미만이면 건너뛰기
    if len(data) < 10:
        continue

    X = data[["충전시간_분"]]
    y = data["충전량(kW)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "충전소명": name,
        "데이터 개수": len(data),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values("R2", ascending=False)
    display(results_df)

print("모델을 실행한 충전소 수:", len(results_df))

# %%
# R² 기준으로 충전소 성능 분류
def classify_r2(r2):
    if r2 < 0:
        return "성능 낮음 (R² < 0)"
    elif r2 < 0.5:
        return "설명력 낮음 (0 ≤ R² < 0.5)"
    else:
        return "설명력 상대적으로 높음 (R² ≥ 0.5)"

results_df["성능 분류"] = results_df["R2"].apply(classify_r2)

# 분류별 충전소 개수
print(results_df["성능 분류"].value_counts())

# 분류별 충전소 목록 확인
display(
    results_df.sort_values(["성능 분류", "R2"])[
        ["충전소명", "데이터 개수", "MAE", "RMSE", "R2", "성능 분류"]
    ]
)

# %%
# R²가 음수인 충전소 13개 확인
low_r2 = results_df[results_df["R2"] < 0].sort_values("R2")

display(low_r2[
    ["충전소명", "데이터 개수", "MAE", "RMSE", "R2"]
])

# %%
# R²가 가장 낮은 충전소 확인
display(
    results_df[results_df["R2"] < 0]
    .sort_values("R2")
    .head(13)
)

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# R²가 가장 낮은 충전소 선택
name = results_df[results_df["R2"] < 0].sort_values("R2").iloc[0]["충전소명"]

data = df_model[df_model["충전소명"] == name][
    ["충전시간_분", "충전량(kW)"]
].dropna().copy()

# 해당 충전소의 선형회귀 직선 계산
X = data[["충전시간_분"]]
y = data["충전량(kW)"]

model_check = LinearRegression()
model_check.fit(X, y)

data["예측값"] = model_check.predict(X)
data["절대오차"] = (data["충전량(kW)"] - data["예측값"]).abs()

# 실제 데이터와 직선 표시
plt.figure(figsize=(8, 5))
plt.scatter(data["충전시간_분"], y, alpha=0.5, label="Actual")
plt.plot(
    data["충전시간_분"],
    data["예측값"],
    color="red",
    label="Linear Regression"
)
plt.xlabel("충전시간(분)")
plt.ylabel("충전량")
plt.title(f"{name} (R² < 0)")
plt.legend()
plt.show()

print("선택한 충전소:", name)

# %%
# 직선에서 가장 멀리 떨어진 데이터 10개 확인
display(
    data.sort_values("절대오차", ascending=False)[
        ["충전시간_분", "충전량(kW)", "예측값", "절대오차"]
    ].head(10)
)

# %%
# 직선에서 가장 멀리 떨어진 데이터의 원본 기록 확인
row_index = 132727

display(
    df_model.loc[
        [row_index],
        ["충전소명", "충전시작시간", "충전종료시간",
         "충전시간_분", "충전량(kW)"]
    ]
)

# %%
print("현재 df_model 데이터 개수:", len(df_model))

# %%
# 1. 이상치가 발견된 충전소 이름 지정
name = "앵봉산가족캠핑장"

# 2. 해당 충전소 데이터만 가져오기
station_data = df_model[
    df_model["충전소명"] == name
].copy()

# 3. 이상치 인덱스 132727만 제외
station_data_clean = station_data.drop(index=132727)

print("기존 데이터 개수:", len(station_data))
print("이상치 제외 후:", len(station_data_clean))

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 입력 변수와 정답 설정
X = station_data_clean[["충전시간_분"]]
y = station_data_clean["충전량(kW)"]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 선형회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mae_after = mean_absolute_error(y_test, y_pred)
rmse_after = np.sqrt(mean_squared_error(y_test, y_pred))
r2_after = r2_score(y_test, y_pred)

print("=== 이상치 제외 후 성능 ===")
print("데이터 개수:", len(station_data_clean))
print("MAE:", round(mae_after, 4))
print("RMSE:", round(rmse_after, 4))
print("R²:", round(r2_after, 4))

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 남산청사 데이터만 가져오기
station_data = df_model[
    df_model["충전소명"] == "남산청사"
][["충전시간_분", "충전량(kW)"]].dropna().copy()

# 모델 학습
X = station_data[["충전시간_분"]]
y = station_data["충전량(kW)"]

model = LinearRegression()
model.fit(X, y)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.scatter(X["충전시간_분"], y, alpha=0.5, label="Actual data")

plt.plot(
    X["충전시간_분"],
    model.predict(X),
    color="red",
    label="Linear regression"
)

plt.xlabel("Charging time (minutes)")
plt.ylabel("Charging amount (kW)")
plt.title("Namsan Office: Charging Time vs Charging Amount")
plt.legend()
plt.grid(True)
plt.show()

# %%
# 남산청사 데이터 불러오기
namsan_data = df_model[
    df_model["충전소명"] == "남산청사"
].copy()

print("남산청사 데이터 개수:", len(namsan_data))

display(namsan_data.head(20))

# %%
# 남산청사 데이터 불러오기
namsan_data = df_model[
    df_model["충전소명"] == "남산청사"
][["충전시간_분", "충전량(kW)"]].dropna().copy()

from sklearn.linear_model import LinearRegression

X = namsan_data[["충전시간_분"]]
y = namsan_data["충전량(kW)"]

# 전체 데이터로 회귀선 생성 (이상치 탐색용)
model_check = LinearRegression()
model_check.fit(X, y)

# 예측값과 오차 계산
namsan_data["예측값"] = model_check.predict(X)
namsan_data["절대오차"] = (
    namsan_data["충전량(kW)"] - namsan_data["예측값"]
).abs()

# 회귀선에서 가장 멀리 떨어진 기록 20개
display(
    namsan_data.sort_values("절대오차", ascending=False).head(20)
)

# %%
# 회귀선에서 멀리 떨어진 기록 4개 확인
check_indices = [136756, 208453, 112689, 74368]

display(
    df_model.loc[
        check_indices,
        ["충전소명", "충전시작시간", "충전종료시간",
         "충전시간_분", "충전량(kW)"]
    ]
)

# %%
# 남산청사 데이터
namsan_data = df_model[
    df_model["충전소명"] == "남산청사"
].copy()

# 인덱스 확인
print("해당 인덱스 존재 여부:", 136756 in namsan_data.index)

# 첫 번째 기록만 제외
namsan_clean = namsan_data.drop(index=136756)

print("제외 전:", len(namsan_data))
print("제외 후:", len(namsan_clean))

# %%
# 앵봉산가족캠핑장 이상치 삭제
df_model = df_model.drop(index=132727, errors="ignore")

# 남산청사 이상치 삭제
df_model = df_model.drop(index=136756, errors="ignore")

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []

for name, data in df_model.groupby("충전소명"):

    data = data[["충전시간_분", "충전량(kW)"]].dropna()

    if len(data) < 10:
        continue

    X = data[["충전시간_분"]]
    y = data["충전량(kW)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "충전소명": name,
        "데이터 개수": len(data),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)

    display(results_df)

print("모델을 실행한 충전소 수:", len(results_df))
print("R² 음수인 충전소 수:", (results_df["R2"] < 0).sum())

# %%
low_r2 = results_df[results_df["R2"] < 0].copy()

low_r2 = low_r2.sort_values("R2")

display(
    low_r2[
        ["충전소명", "데이터 개수", "MAE", "RMSE", "R2"]
    ]
)

print("R²가 0보다 작은 충전소 수:", len(low_r2))

# %%
import matplotlib.pyplot as plt

name = "50+동부캠퍼스"

station_data = df_model[
    df_model["충전소명"].astype(str).str.strip() == name
].copy()

print("데이터 개수:", len(station_data))

plt.figure(figsize=(8, 5))
plt.scatter(
    station_data["충전시간_분"],
    station_data["충전량(kW)"],
    alpha=0.5
)

plt.xlabel("충전시간(분)")
plt.ylabel("충전량(kW)")
plt.title(f"{name} 충전시간과 충전량")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
name = "50+동부캠퍼스"

outliers = df_model[
    (df_model["충전소명"].astype(str).str.strip() == name) &
    (df_model["충전량(kW)"] <= 20) &
    (df_model["충전시간_분"] >= 300)
]

display(
    outliers[
        ["충전소명", "충전시작시간", "충전종료시간",
         "충전시간_분", "충전량(kW)"]
    ]
)

print("해당 데이터 인덱스:", outliers.index.tolist())

# %%
# 50+동부캠퍼스 이상치 2개 제거
df_model = df_model.drop(index=[139362, 160337], errors="ignore")

# 불암산공영주차장 삭제
df_model = df_model.drop(index=[47874], errors="ignore")

# 여의서로 노상공영주차장(가로등) 삭제
df_model = df_model.drop(index=[20577], errors="ignore")

# 노원구청 이상치 2개 제거
df_model = df_model.drop(index=[91016, 90663], errors="ignore")

## 한마음공영주차장 제거
df_model = df_model.drop(index=[211049], errors="ignore")

#  어린이대공원 구의문 주차장 복합충전소 이상치 2개 삭제
df_model = df_model.drop(index=[90474, 157271], errors="ignore")

# 교통공원길 충전시간 이상치 후보 제거
df_model = df_model.drop(index=[138608], errors="ignore")

# 월계복지문화센터 이상치 데이터 제거
df_model = df_model.drop(index=[30478], errors="ignore")

# 고척스카이돔 이상치 데이터 제거
df_model = df_model.drop(index=[143415], errors="ignore")

# 마포유수지 공영주차장 이상치 데이터 제거
df_model = df_model.drop(index=[193769], errors="ignore")



# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []

# 충전소별 선형회귀 재평가
for name, data in df_model.groupby("충전소명"):

    data = data[["충전시간_분", "충전량(kW)"]].dropna()

    # 데이터가 너무 적거나 충전시간이 모두 같으면 제외
    if len(data) < 10 or data["충전시간_분"].nunique() < 2:
        continue

    X = data[["충전시간_분"]]
    y = data["충전량(kW)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "충전소명": name,
        "데이터 수": len(data),
        "실제 평균 충전량": y_test.mean(),
        "예측 평균 충전량": y_pred.mean(),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

# R²가 0보다 작은 충전소만 추출
negative_r2 = results_df[results_df["R2"] < 0].sort_values("R2")

display(negative_r2.round(4))



# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []

# 충전소별 모델 재학습
for name, data in df_model.groupby("충전소명"):

    data = data[["충전시간_분", "충전량(kW)"]].dropna()

    if len(data) < 10 or data["충전시간_분"].nunique() < 2:
        continue

    X = data[["충전시간_분"]]
    y = data["충전량(kW)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "충전소명": name,
        "데이터 수": len(data),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

# 결과 갱신
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R2").reset_index(drop=True)

# R²가 음수인 충전소만 출력
negative_r2 = results_df[results_df["R2"] > 0]

display(negative_r2.round(4))

# %%
# 처음에 R²가 음수였던 13개 충전소
target_stations = [
    "앵봉산가족캠핑장",
    "남산청사",
    "한마음공영주차장",
    "마포유수지 공영주차장",
    "국립한글박물관 지하주차장(주차장 임시폐쇄)",
    "여의서로 노상공영주차장(가로등)",
    "불암산공영주차장",
    "노원구청",
    "어린이대공원 구의문 주차장 복합충전소",
    "교통공원길 공영주차장 전기차 충전소",
    "50+동부캠퍼스",
    "월계복지문화센터",
    "고척스카이돔"
]

# 최신 df_model로 해당 충전소들만 재평가
comparison_results = []

for name in target_stations:
    data = df_model[
        df_model["충전소명"].astype(str).str.strip() == name
    ][["충전시간_분", "충전량(kW)"]].dropna()

    if len(data) < 10 or data["충전시간_분"].nunique() < 2:
        print(f"데이터 부족 또는 충전시간 변동 없음: {name}")
        continue

    X = data[["충전시간_분"]]
    y = data["충전량(kW)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    comparison_results.append({
        "충전소명": name,
        "데이터 수": len(data),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

comparison_df = pd.DataFrame(comparison_results)

display(comparison_df.round(4))

# %%
print("df_model 전체 데이터:", len(df_model), "개")

# %%



