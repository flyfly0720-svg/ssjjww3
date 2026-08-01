# 비만·운동과 암: Cox 회귀분석 + YAP/TAZ 미분방정식 모델링

## 폴더 구성
- `app.py` — Streamlit 앱 본체 (4개 탭)
- `hn24_cox_data.csv` — 국민건강영양조사(HN24, 2024) 원자료에서 분석에 필요한
  변수만 정제한 경량 데이터 (BMI, 유산소 신체활동 실천율, 공복혈당, 암 진단 시점 등)
- `requirements.txt` — 필요 패키지 목록

## 실행 방법
같은 폴더 안에 네 파일(`app.py`, `hn24_cox_data.csv`, `requirements.txt`, `README.md`)을
반드시 함께 두어야 합니다. `app.py`는 자기 자신의 파일 위치를 기준으로
`hn24_cox_data.csv`를 절대경로로 찾으므로, 어느 디렉토리에서 실행하든
CSV가 app.py와 "같은 폴더"에만 있으면 정상 작동합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

실행 후 브라우저에서 자동으로 열리는 주소(보통 http://localhost:8501)로 접속하면 됩니다.

### Streamlit Community Cloud에 배포할 때 자주 발생하는 FileNotFoundError
아래 항목을 순서대로 확인하세요.
1. **GitHub 저장소에 CSV가 실제로 올라갔는지 확인** — `.gitignore` 파일에
   `*.csv`나 `data/` 같은 규칙이 있으면 CSV가 커밋에서 자동 제외됩니다.
   저장소 웹페이지에서 `hn24_cox_data.csv`가 눈으로 보이는지 확인하세요.
2. **폴더 위치 확인** — `app.py`와 `hn24_cox_data.csv`가 저장소 안에서
   정확히 같은 폴더(같은 depth)에 있어야 합니다. 하나는 루트에, 하나는
   하위 폴더에 있으면 안 됩니다.
3. **Streamlit Cloud의 "Main file path" 설정 확인** — 앱 배포 설정에서
   지정한 진입점 파일 경로가 실제 `app.py` 위치와 일치하는지 확인하세요.
4. 위를 다 확인했는데도 안 되면, "Manage app" → 하단 로그 패널에서
   전체 traceback을 확인해 정확한 원인을 특정하세요.

## 4개 탭 구성
1. **데이터 & Cox 회귀분석** — 실제 관측자료 기반, 비만·운동이 암 진단까지 걸리는
   시간에 미치는 영향을 위험비(HR)로 추정
2. **YAP/TAZ 미분방정식 모형** — Akt(비만)-AMPK(운동)-YAP/TAZ-암세포 동역학을
   상미분방정식으로 시뮬레이션, 모든 파라미터를 슬라이더로 조절 가능
3. **모형 진단(디버깅)** — ODE 수치해 이상 여부를 자동 진단 + Cox 모형과 ODE 모형의
   예측이 어긋나는 지점을 역추적
4. **민감도 분석** — ODE 파라미터가 가정치임을 보완하기 위해, 파라미터를 체계적으로
   바꾸며 결과의 안정성을 검증

## 데이터 재생성이 필요한 경우
`hn24_cox_data.csv`는 질병관리청 국민건강영양조사 원시자료(hn24_all.sas7bdat)에서
아래 로직으로 파생시킨 것입니다. 원본 SAS 파일을 다시 가공하려면:

```python
import pyreadstat, pandas as pd, numpy as np

cols = ["ID","sex","age","DC01_dg","DC01_ag","HE_BMI","pa_aerobic","HE_glu"]
df, meta = pyreadstat.read_sas7bdat("hn24_all.sas7bdat", usecols=cols)

df_adult = df[df["age"] >= 19].copy()
df_adult["cancer"] = df_adult["DC01_dg"].replace({8: np.nan, 9: np.nan})
df_adult["female"] = df_adult["sex"].map({1: 0, 2: 1})
df_adult["event"] = df_adult["cancer"]
df_adult["time"] = np.where(df_adult["cancer"] == 1, df_adult["DC01_ag"], df_adult["age"])

keep = ["ID","time","event","HE_BMI","pa_aerobic","HE_glu","age","female"]
df_final = df_adult[keep].dropna().reset_index(drop=True)
df_final.columns = ["ID","time","event","BMI","aerobic_pa","fasting_glucose","age","female"]
df_final.to_csv("hn24_cox_data.csv", index=False, encoding="utf-8-sig")
```

## 방법론적 한계 (보고서 작성 시 반드시 명시)
- 원자료는 **단면조사**이므로 BMI·운동·혈당은 2024년 현재 시점 측정값이고,
  암 진단 시점은 과거이므로 엄밀한 인과관계를 증명하지 않습니다.
- YAP/TAZ 미분방정식의 파라미터(α, β, γ, r, K, δ)는 문헌에서 추정된 값이 아니라
  모형의 정성적 동역학을 보여주기 위한 **가정치**입니다. 4번 탭의 민감도 분석
  결과를 반드시 함께 제시해 이 한계를 보완하시길 권장합니다.
