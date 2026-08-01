
# -*- coding: utf-8 -*-
"""
비만·운동이 암 발생에 미치는 영향: Cox 회귀분석 + YAP/TAZ 미분방정식 모델링
국민건강영양조사 제9기 2차년도(2024, HN24) 원시자료 기반

[전체 구성]
탭 1. 데이터 & Cox 비례위험 회귀분석
     - 실제 관측자료로 "비만(BMI)·운동(유산소 신체활동)이 암 진단까지의 시간에
       미치는 영향"을 위험비(Hazard Ratio)로 추정
탭 2. YAP/TAZ 미분방정식 모형
     - Akt(비만 신호) - AMPK(운동 신호) - YAP/TAZ(온코진) - 암세포 증식의
       동역학을 상미분방정식(ODE)으로 시뮬레이션. 파라미터는 전부 슬라이더로 조절 가능
탭 3. 모형 진단 (디버깅)
     - ODE 수치해가 발산/이상값을 보일 때 원인을 자동으로 진단
     - Cox 모형(통계적 결과)과 ODE 모형(기전적 예측)이 서로 어긋나는 지점을 찾아
       "왜 두 모형이 다른 답을 주는가"를 역으로 추적
탭 4. 민감도 분석
     - ODE 파라미터는 전부 가정치이므로, 파라미터를 체계적으로 흔들어 결과가
       얼마나 민감하게 바뀌는지 보여줌으로써 "가정치 기반 모형"이라는 한계를 보완
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

st.set_page_config(
    page_title="비만·운동과 암: Cox 회귀 + YAP/TAZ 모델링",
    layout="wide",
)

st.title("비만·운동이 암 발생에 미치는 영향")
st.caption("국민건강영양조사(HN24, 2024) 자료 기반 Cox 회귀분석 · YAP/TAZ 미분방정식 모델링")

# ------------------------------------------------------------------
# 데이터 로드 (캐싱하여 재실행 시 빠르게)
#
# 주의: Streamlit Cloud 등 배포 환경에서는 앱이 실행되는 "현재 작업 디렉토리"가
# 저장소 루트가 아닐 수 있습니다. 그래서 "hn24_cox_data.csv" 같은 상대경로만
# 쓰면 FileNotFoundError가 날 수 있습니다. 이를 피하기 위해 이 스크립트
# 파일(app.py) 자신의 위치를 기준으로 절대경로를 만들어 사용합니다.
# 즉 app.py를 어느 폴더에서 실행하든, CSV는 반드시 app.py와 "같은 폴더"에서
# 찾도록 고정합니다.
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "hn24_cox_data.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # 배포 환경에서 흔한 원인들을 구체적으로 안내합니다.
        st.error(
            "데이터 파일(hn24_cox_data.csv)을 찾을 수 없습니다.\n\n"
            f"찾고 있는 경로: `{path}`\n\n"
            "**확인해보세요:**\n"
            "1. `hn24_cox_data.csv`가 `app.py`와 **같은 폴더**에 있는지 (하위 폴더에 있으면 안 됩니다)\n"
            "2. GitHub 저장소에 이 CSV 파일이 실제로 커밋·푸시되었는지 "
            "(`.gitignore`에 `*.csv`가 있으면 제외될 수 있습니다)\n"
            "3. 파일 용량이 100MB를 넘지 않는지 (GitHub 기본 업로드 한도)\n"
            "4. 파일명 대소문자가 정확히 일치하는지 (Linux 서버는 대소문자를 구분합니다)"
        )
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()


df = load_data(DATA_PATH)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. 데이터 & Cox 회귀분석",
    "2. YAP/TAZ 미분방정식 모형",
    "3. 모형 진단 (디버깅)",
    "4. 민감도 분석",
])

# ==================================================================
# 탭 1. 데이터 & Cox 비례위험 회귀분석
# ==================================================================
with tab1:
    st.header("Cox 비례위험 회귀분석")

    st.markdown("""
    **왜 로지스틱이 아니라 Cox 회귀인가?**
    로지스틱 회귀는 "암 진단 여부(0/1)"만 보고 **언제** 진단받았는지, 그리고
    관찰이 **언제까지** 이어졌는지(=아직 안 걸렸다는 정보의 신뢰도)를 버립니다.
    예를 들어 79세까지 안 걸린 사람과 20세인데 아직 안 걸린 사람은 로지스틱에서
    똑같이 "0"이지만, 위험도에 대한 정보량은 전혀 다릅니다.
    Cox 모형은 이 **시간 정보**를 살려서, 비만·운동 같은 변수가 "암이 발생하기까지
    걸리는 시간"에 미치는 영향을 위험비(Hazard Ratio, HR)로 추정합니다.
    """)

    with st.expander("변수 구성 방법 (원자료 → 분석용 변수)", expanded=False):
        st.markdown("""
        | 분석 변수 | 원자료 변수 | 설명 |
        |---|---|---|
        | `time` (사건 발생/관찰 종료 시간) | `DC01_ag`(진단자) 또는 `age`(비진단자) | 암 진단자는 **진단 시 나이**, 비진단자는 **현재 나이**(중도절단, censoring) |
        | `event` (사건 발생 여부) | `DC01_dg` | 1=암 진단됨(사건 발생), 0=미진단(중도절단) |
        | `BMI` | `HE_BMI` | 체질량지수(연속) |
        | `aerobic_pa` | `pa_aerobic` | 유산소 신체활동 실천율(0=미실천, 1=실천) |
        | `fasting_glucose` | `HE_glu` | 공복혈당(mg/dL) |
        | `female` | `sex` | 0=남성, 1=여성 |

        분석 대상은 만 19세 이상 성인이며, 8(비해당)·9(모름/무응답) 코드와
        결측치는 모두 제외했습니다. 최종 표본 수는 **{}명**입니다.
        """.format(len(df)))

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 분석 표본", f"{len(df):,}명")
    col2.metric("암 진단(사건 발생)", f"{int(df['event'].sum()):,}명")
    col3.metric("중도절단(미진단)", f"{int((df['event']==0).sum()):,}명")

    st.subheader("공변량 선택")
    covariate_options = {
        "BMI (체질량지수)": "BMI",
        "aerobic_pa (유산소 신체활동 실천율)": "aerobic_pa",
        "fasting_glucose (공복혈당)": "fasting_glucose",
        "female (성별, 0=남 1=여)": "female",
    }
    selected_labels = st.multiselect(
        "Cox 모형에 포함할 변수를 선택하세요",
        options=list(covariate_options.keys()),
        default=list(covariate_options.keys()),
    )
    selected_cols = [covariate_options[l] for l in selected_labels]

    if len(selected_cols) == 0:
        st.warning("최소 1개 이상의 변수를 선택해주세요.")
    else:
        cph = CoxPHFitter()
        fit_df = df[["time", "event"] + selected_cols]
        cph.fit(fit_df, duration_col="time", event_col="event")

        st.session_state["cph_model"] = cph
        st.session_state["cph_data"] = fit_df

        summary = cph.summary.copy()
        summary_display = summary[["coef", "exp(coef)", "se(coef)",
                                     "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
        summary_display.columns = ["회귀계수", "위험비(HR)", "표준오차",
                                     "HR 95% 하한", "HR 95% 상한", "p-value"]
        st.dataframe(summary_display.style.format("{:.4f}"), use_container_width=True)

        st.markdown(f"""
        **모형 적합도**: Concordance index = `{cph.concordance_index_:.3f}`
        (0.5=무작위 예측 수준, 1.0=완벽한 예측. 통상 0.6~0.7이면 실무적으로 의미 있는 수준으로 봅니다.)
        """)

        # 위험비 forest plot
        fig = go.Figure()
        y_labels = summary_display.index.tolist()
        hr = summary_display["위험비(HR)"].values
        lo = summary_display["HR 95% 하한"].values
        hi = summary_display["HR 95% 상한"].values

        fig.add_trace(go.Scatter(
            x=hr, y=y_labels,
            error_x=dict(type="data", symmetric=False,
                          array=hi - hr, arrayminus=hr - lo),
            mode="markers", marker=dict(size=12, color="#4C72B0"),
            name="위험비(HR)"
        ))
        fig.add_vline(x=1, line_dash="dash", line_color="gray",
                       annotation_text="HR=1 (영향 없음)")
        fig.update_layout(
            title="변수별 위험비(HR) 및 95% 신뢰구간",
            xaxis_title="Hazard Ratio",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **해석 시 유의할 점**: `aerobic_pa`(유산소 신체활동)의 위험비가 1보다 크게 나올 수 있는데,
        이는 "운동이 암을 유발한다"는 뜻이 아니라 **역인과관계(reverse causation)**일 가능성이 높습니다.
        이 자료는 단면조사라 운동량과 BMI·혈당을 **암 진단 이후 시점**(진단받고 나서 건강관리를 위해
        운동을 시작했거나, 반대로 컨디션 저하로 운동을 줄인 상태)에 측정했을 수 있습니다.
        이 문제는 3번 탭 "모형 진단"에서 더 자세히 다룹니다.
        """)

# ==================================================================
# 탭 2. YAP/TAZ 미분방정식 모형
# ==================================================================
with tab2:
    st.header("YAP/TAZ 동역학 미분방정식 모형")

    st.markdown("""
    Cox 회귀분석이 "비만·운동과 암 발생 시점 사이의 **통계적** 연관성"을 보여준다면,
    이 미분방정식 모형은 그 연관성이 세포 내에서 **어떤 분자 기전**을 거쳐 나타날 수
    있는지를 시뮬레이션합니다. 핵심 축은 **YAP/TAZ**라는 전사보조인자(transcriptional
    co-activator)입니다. YAP/TAZ는 세포 증식·생존 신호를 핵 안으로 전달하는 스위치
    역할을 하며, 정상적으로는 Hippo 신호전달경로에 의해 억제되어 있다가, 특정 자극이
    있으면 핵으로 이동해 암 관련 유전자 발현을 촉진합니다.
    """)

    st.subheader("모형 구조: 왜 이렇게 식을 세웠는가")

    st.latex(r"\frac{dY}{dt} = \alpha \cdot A - \beta \cdot M \cdot Y - \gamma \cdot Y")
    st.latex(r"\frac{dC}{dt} = r \cdot C \left(1 - \frac{C}{K}\right) + \delta \cdot Y \cdot C")

    with st.expander("1번째 식: YAP/TAZ 농도 변화(dY/dt)를 이렇게 세운 이유", expanded=True):
        st.markdown(r"""
        $Y(t)$는 시간 $t$에서 핵 내 활성형 YAP/TAZ의 농도입니다. 이 식은 세 개의 항이
        더해진 형태인데, 각 항이 왜 필요한지 하나씩 보면:

        - **생성 항 $+\alpha \cdot A$**: 비만·고혈당 상태에서는 세포 내 Akt 신호전달이
          활성화됩니다. Akt는 Hippo 경로의 핵심 억제 인자인 LATS1/2를 억제해서,
          결과적으로 YAP/TAZ가 인산화되지 않은 채 핵으로 이동하는 것을 돕습니다.
          즉 $A$(Akt 활성도)가 클수록 YAP/TAZ가 더 많이, 더 빠르게 생성(정확히는 핵 내
          축적)된다는 뜻이라 **곱셈 항 $\alpha A$**로 표현했습니다. 여기서 $\alpha$는
          "Akt 신호 한 단위가 YAP/TAZ 축적 속도를 얼마나 높이는가"를 나타내는 비례상수입니다.

        - **소실 항 $-\beta \cdot M \cdot Y$**: 운동을 하면 AMPK(AMP-활성화 단백질 인산화효소)가
          활성화되는데, AMPK는 YAP를 직접 인산화시켜 세포질에 붙잡아 두거나 분해를 촉진합니다.
          이 소실 속도는 **AMPK 활성도($M$)와 현재 YAP/TAZ 농도($Y$) 둘 다에 비례**해야
          합니다 — YAP/TAZ가 아예 없으면(Y=0) AMPK가 아무리 활성화돼도 없앨 대상이 없으므로
          소실량도 0이어야 하기 때문입니다. 그래서 $M$과 $Y$의 **곱**으로 씁니다.

        - **자연 분해 항 $-\gamma \cdot Y$**: 운동 자극이 전혀 없어도(M=0) 단백질은
          세포 내에서 유비퀴틴-프로테아좀 경로를 통해 계속 분해됩니다. 이 "기저 분해율"을
          $\gamma$로 표현했고, 역시 남아있는 농도 $Y$에 비례합니다(1차 반응 속도론,
          즉 분해 속도가 현재 양에 비례한다는 화학반응속도론의 표준적 가정입니다).

        결과적으로 $Y$는 "비만 신호로 계속 채워지고, 운동 신호와 자연 분해로 계속
        빠지는" 저수지처럼 움직입니다.
        """)

    with st.expander("2번째 식: 암세포 수 변화(dC/dt)를 이렇게 세운 이유", expanded=True):
        st.markdown(r"""
        $C(t)$는 초기 변이 암세포의 수입니다. 이 식도 두 항으로 나뉩니다.

        - **로지스틱 증식 항 $r \cdot C \left(1 - \dfrac{C}{K}\right)$**: 암세포는
          자원(영양분·산소·공간)이 무한하지 않은 한 무한정 늘어날 수 없습니다.
          $r$은 자원이 충분할 때의 최대 증식률이고, $K$는 그 조직이 감당할 수 있는
          최대 세포 수(환경 수용력, carrying capacity)입니다. $C$가 $K$에 가까워질수록
          $\left(1-\frac{C}{K}\right)$가 0에 가까워져서 증식이 자연스럽게 둔화됩니다.
          이건 생태학의 표준 로지스틱 성장모형과 동일한 논리입니다.

        - **YAP/TAZ 매개 가속 항 $+\delta \cdot Y \cdot C$**: YAP/TAZ가 핵 안에서
          활성화되면 세포주기 촉진 유전자(예: CCND1) 발현을 늘려 암세포 증식을
          가속시킵니다. 이 효과는 **현재 YAP/TAZ 농도($Y$)와 현재 암세포 수($C$)의 곱**에
          비례해야 합니다 — YAP/TAZ가 아무리 높아도 암세포가 0개면 가속될 대상이
          없고(C=0이면 항 전체가 0), 반대로 암세포가 많아도 YAP/TAZ가 0이면 추가
          가속이 없어야(Y=0이면 항 전체가 0) 하기 때문에 곱셈 구조가 맞습니다.
          $\delta$는 "YAP/TAZ 농도 한 단위가 암세포 증식을 얼마나 더 가속하는가"를
          나타내는 결합 상수입니다.
        """)

    st.subheader("파라미터 설정")
    st.caption("""
    ⚠️ 아래 파라미터는 문헌에서 직접 추정된 값이 아니라, 모형의 정성적 동역학(비만 신호가
    강해지면 암세포가 늘고, 운동 신호가 강해지면 억제된다는 방향성)을 보여주기 위한
    **가정치(assumed values)**입니다. 실제 크기가 아니라 **상대적 변화의 패턴**에
    초점을 맞춰 해석하시고, 4번 탭의 민감도 분석과 함께 보시길 권장합니다.
    """)

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        st.markdown("**YAP/TAZ 동역학 파라미터**")
        alpha = st.slider(
            "α (알파) — Akt→YAP/TAZ 생성 속도상수", 0.0, 2.0, 0.5, 0.05,
            help="비만·고혈당으로 인한 Akt 신호 1단위가 YAP/TAZ를 얼마나 빨리 축적시키는지. "
                 "값이 클수록 같은 비만 정도(A)에서도 YAP/TAZ가 더 빠르게 쌓입니다."
        )
        beta = st.slider(
            "β (베타) — AMPK→YAP/TAZ 분해 속도상수", 0.0, 2.0, 0.4, 0.05,
            help="운동으로 인한 AMPK 신호 1단위가 YAP/TAZ를 얼마나 빨리 제거하는지. "
                 "값이 클수록 같은 운동량(M)에서도 YAP/TAZ가 더 빠르게 억제됩니다."
        )
        gamma = st.slider(
            "γ (감마) — YAP/TAZ 자연 분해율", 0.0, 1.0, 0.2, 0.01,
            help="운동 자극과 무관하게 세포 내에서 기본적으로 일어나는 YAP/TAZ 단백질 "
                 "분해 속도(유비퀴틴-프로테아좀 경로 등). 값이 클수록 개입 없이도 "
                 "YAP/TAZ가 빨리 없어집니다."
        )
        A_param = st.slider(
            "A — 비만/고혈당에 의한 Akt 활성 상수", 0.0, 5.0, 2.0, 0.1,
            help="현재 대사 상태(체지방, 인슐린 저항성 등)에 따른 Akt 활성화 정도를 "
                 "나타내는 외생 변수입니다. BMI나 공복혈당이 높을수록 이 값이 커진다고 "
                 "가정할 수 있습니다."
        )

    with pcol2:
        st.markdown("**암세포 증식 파라미터**")
        r_param = st.slider(
            "r — 암세포 최대 증식률", 0.0, 2.0, 0.3, 0.01,
            help="자원이 충분할 때 암세포가 스스로 증식하는 최대 속도. YAP/TAZ와 무관한, "
                 "암세포 고유의 증식 능력을 나타냅니다."
        )
        K_param = st.slider(
            "K — 환경 수용력(최대 세포 수)", 10.0, 500.0, 100.0, 10.0,
            help="해당 조직 환경이 물리적으로 감당 가능한 암세포 수의 상한선. 혈액 공급, "
                 "공간적 제약 등에 의해 결정됩니다."
        )
        delta = st.slider(
            "δ (델타) — YAP/TAZ의 암세포 증식 가속 계수", 0.0, 0.1, 0.02, 0.001,
            help="YAP/TAZ 농도가 암세포 증식 속도를 얼마나 가속시키는지. 이 값이 0이면 "
                 "YAP/TAZ와 무관하게 암세포는 순수 로지스틱 증식만 합니다."
        )
        M_param = st.slider(
            "M — 운동에 의한 AMPK 활성 상수", 0.0, 5.0, 1.0, 0.1,
            help="현재 신체활동 수준에 따른 AMPK 활성화 정도를 나타내는 외생 변수입니다. "
                 "유산소 운동 실천율이 높을수록 이 값이 커진다고 가정할 수 있습니다."
        )

    icol1, icol2, icol3 = st.columns(3)
    with icol1:
        Y0 = st.number_input("Y(0) — YAP/TAZ 초기 농도", 0.0, 20.0, 1.0, 0.1)
    with icol2:
        C0 = st.number_input("C(0) — 암세포 초기 개수", 0.1, 50.0, 1.0, 0.1)
    with icol3:
        t_max = st.number_input("시뮬레이션 시간 범위 (t_max)", 10, 200, 100, 10)

    def yap_taz_model(t, state, alpha, beta, gamma, A, r, K, delta, M):
        Y, C = state
        dYdt = alpha * A - beta * M * Y - gamma * Y
        dCdt = r * C * (1 - C / K) + delta * Y * C
        return [dYdt, dCdt]

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 500)

    sol = solve_ivp(
        yap_taz_model, t_span, [Y0, C0],
        args=(alpha, beta, gamma, A_param, r_param, K_param, delta, M_param),
        t_eval=t_eval, method="RK45", dense_output=True,
    )

    st.session_state["ode_solution"] = sol
    st.session_state["ode_params"] = dict(
        alpha=alpha, beta=beta, gamma=gamma, A=A_param,
        r=r_param, K=K_param, delta=delta, M=M_param, Y0=Y0, C0=C0, t_max=t_max
    )

    st.subheader("시뮬레이션 결과")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name="Y(t): YAP/TAZ 농도",
                                line=dict(color="#DD8452", width=3)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name="C(t): 암세포 수",
                                line=dict(color="#C44E52", width=3)), secondary_y=True)
    fig2.update_xaxes(title_text="시간 t")
    fig2.update_yaxes(title_text="YAP/TAZ 농도 Y(t)", secondary_y=False)
    fig2.update_yaxes(title_text="암세포 수 C(t)", secondary_y=True)
    fig2.update_layout(title="시간에 따른 YAP/TAZ 농도와 암세포 수 변화",
                        template="plotly_white", height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    **정상상태(steady-state) 이론값과 비교**: $M=0$이면 $Y^* = \\dfrac{{\\alpha A}}{{\\gamma}}$,
    $M>0$이면 $Y^* = \\dfrac{{\\alpha A}}{{\\beta M + \\gamma}}$ 로 수렴합니다.

    현재 파라미터 기준 이론적 정상상태: $Y^* = {(alpha*A_param)/(beta*M_param+gamma):.3f}$
    (시뮬레이션 마지막 값: {sol.y[0][-1]:.3f})
    """)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sol.y[0], y=sol.y[1], mode="lines",
                                line=dict(color="#55A868", width=2),
                                name="궤적"))
    fig3.add_trace(go.Scatter(x=[sol.y[0][0]], y=[sol.y[1][0]], mode="markers",
                                marker=dict(size=12, color="blue"), name="시작점"))
    fig3.add_trace(go.Scatter(x=[sol.y[0][-1]], y=[sol.y[1][-1]], mode="markers",
                                marker=dict(size=12, color="red"), name="종료점"))
    fig3.update_layout(title="위상평면(Phase Plane): Y(YAP/TAZ)와 C(암세포 수)의 관계",
                        xaxis_title="Y(t): YAP/TAZ 농도", yaxis_title="C(t): 암세포 수",
                        template="plotly_white", height=450)
    st.plotly_chart(fig3, use_container_width=True)

# ==================================================================
# 탭 3. 모형 진단 (디버깅)
# ==================================================================
with tab3:
    st.header("모형 진단: 결과에서 역으로 문제를 추적하기")

    st.markdown("""
    이 탭은 두 가지를 진단합니다.
    **(A)** ODE 수치해 자체에 문제(발산, 이상값, 음수화 등)가 있는지 자동으로 점검하고 원인을 역추적합니다.
    **(B)** Cox 모형(통계적 관측)과 ODE 모형(기전적 예측)이 서로 다른 결론을 내릴 때, 그 불일치가
    어디서 비롯되는지 짚어봅니다.
    """)

    st.subheader("A. ODE 수치해 자동 진단")

    if "ode_solution" not in st.session_state:
        st.info("먼저 2번 탭에서 미분방정식 시뮬레이션을 실행해주세요.")
    else:
        sol = st.session_state["ode_solution"]
        p = st.session_state["ode_params"]

        Y_arr, C_arr = sol.y[0], sol.y[1]
        issues = []

        # 진단 1: 적분 자체가 실패했는가
        if not sol.success:
            issues.append((
                "🔴 심각",
                "ODE 적분 실패 (solve_ivp success=False)",
                "왜 발생하는가: 스텝 크기가 시스템의 '강성(stiffness)'을 따라가지 못하면 "
                "적분기가 중간에 실패합니다. 강성은 두 변수(Y, C)의 변화 속도 차이가 클 때 "
                "생기는데, 예를 들어 β·M+γ(YAP/TAZ 소실률)가 r(암세포 증식률)보다 훨씬 크면 "
                "Y는 매우 빨리 변하고 C는 천천히 변해서 같은 스텝 크기로 둘 다 정확히 "
                "추적하기 어려워집니다.",
                "해결 방향: RK45 대신 강성 문제에 강한 'Radau'나 'BDF' 방법으로 바꾸거나, "
                "max_step을 더 작게 지정해보세요."
            ))

        # 진단 2: 암세포 수가 음수가 되었는가
        if (C_arr < 0).any():
            first_neg_idx = np.argmax(C_arr < 0)
            issues.append((
                "🔴 심각",
                f"암세포 수 C(t)가 t≈{sol.t[first_neg_idx]:.1f}에서 음수로 전환됨",
                "왜 발생하는가: 로지스틱 항 r·C·(1-C/K)는 C>K일 때 자동으로 음의 되먹임을 "
                "만들어 C를 K 쪽으로 되돌리지만, δ·Y·C 항의 가속이 너무 강하면(δ 또는 Y가 큼) "
                "수치적분 스텝 사이에서 C가 K를 크게 뛰어넘었다가 다음 스텝에서 과도하게 "
                "꺾여 음수로 넘어갈 수 있습니다. 생물학적으로 세포 수는 절대 음수가 될 수 "
                "없으므로, 이는 파라미터 조합이 수치적으로 불안정하다는 신호입니다.",
                f"해결 방향: δ(현재 {p['delta']:.4f}) 또는 초기 YAP/TAZ 농도 Y(0)를 낮추거나, "
                "시뮬레이션 시간 간격을 촘촘히 해보세요 (t_eval 포인트 수를 늘리는 방법도 있습니다)."
            ))

        # 진단 3: YAP/TAZ 또는 암세포 수가 비정상적으로 발산하는가 (K의 10배 초과 등)
        if C_arr.max() > p["K"] * 5:
            issues.append((
                "🟠 주의",
                f"암세포 수가 환경 수용력(K={p['K']:.0f})의 5배 이상으로 발산 (최대 {C_arr.max():.1f})",
                "왜 발생하는가: δ·Y·C 항은 로지스틱 항의 '음의 되먹임'을 갖고 있지 않습니다. "
                "즉 Y가 충분히 크면 δ·Y가 사실상의 새로운 증식률처럼 작동해서, "
                "전체 증식률(r + δ·Y)이 로지스틱 포화 항 (1-C/K)의 억제력을 압도할 수 있습니다. "
                "이 경우 C는 K를 넘어서도 계속 증가하는 비현실적인 궤적을 그립니다.",
                "해결 방향: 이는 모형 설계 자체의 한계를 보여줍니다 — YAP/TAZ 가속항에도 "
                "포화 구조를 추가하거나(예: δ·Y·C·(1-C/K)), α나 A를 낮춰 YAP/TAZ가 "
                "과도하게 축적되지 않도록 조정해보세요."
            ))

        # 진단 4: YAP/TAZ가 정상상태에 도달하지 못했는가 (t_max가 너무 짧음)
        Y_ss_theory = (p["alpha"] * p["A"]) / (p["beta"] * p["M"] + p["gamma"])
        Y_final = Y_arr[-1]
        if abs(Y_final - Y_ss_theory) / (abs(Y_ss_theory) + 1e-9) > 0.05:
            issues.append((
                "🟡 정보",
                f"시뮬레이션 종료 시점에서도 이론적 정상상태에 도달하지 못함 "
                f"(이론값 {Y_ss_theory:.3f} vs 실제 {Y_final:.3f})",
                "왜 발생하는가: 정상상태로 수렴하는 속도(완화 시간, relaxation time)는 "
                "1/(β·M+γ)에 비례합니다. 이 값이 t_max보다 크면(즉 β·M+γ가 작으면) "
                "시뮬레이션이 끝나는 시점까지도 평형에 도달하지 못한 채로 관찰이 종료됩니다.",
                f"해결 방향: t_max(현재 {p['t_max']})를 늘려서 1/(β·M+γ) = "
                f"{1/(p['beta']*p['M']+p['gamma']+1e-9):.1f} 이상 확보하거나, "
                "'아직 평형에 도달하지 않은 과도상태(transient)를 보고 있다'는 점을 "
                "해석에 명시하세요."
            ))

        # 진단 5: 파라미터가 전부 0에 가까워 모형이 사실상 무의미한가
        if p["alpha"] < 0.01 or p["delta"] < 0.001:
            issues.append((
                "🟡 정보",
                "α 또는 δ가 0에 매우 가까움 — 비만 신호나 YAP/TAZ의 암세포 가속 효과가 "
                "사실상 꺼져 있는 상태",
                "왜 발생하는가: α=0이면 Akt 신호가 YAP/TAZ 생성에 전혀 기여하지 못하고, "
                "δ=0이면 YAP/TAZ가 아무리 쌓여도 암세포 증식에 영향을 주지 못합니다. "
                "이 경우 모형은 사실상 '비만-운동-YAP/TAZ-암' 가설을 검증하는 게 아니라 "
                "독립적인 두 개의 방정식(단순 지수 감쇠 + 단순 로지스틱 증식)을 따로 "
                "풀고 있는 셈입니다.",
                "해결 방향: 이 파라미터 조합에서 얻은 결과를 '비만·운동 효과가 있다'는 "
                "근거로 쓰지 않도록 주의하세요. 의도적으로 α나 δ를 0으로 둔 것이라면 "
                "이는 4번 탭의 민감도 분석에서 '기준선(baseline)'으로 쓰기에 적절합니다."
            ))

        if len(issues) == 0:
            st.success("현재 파라미터 조합에서 수치적 이상 징후가 발견되지 않았습니다. "
                       "ODE 해가 안정적으로 수렴했고, 생물학적으로 타당한 범위 내에 있습니다.")
        else:
            for severity, title, why, fix in issues:
                with st.expander(f"{severity} {title}", expanded=True):
                    st.markdown(f"**왜 이런 결과가 나왔는가**\n\n{why}")
                    st.markdown(f"**어떻게 조정해볼 수 있는가**\n\n{fix}")

        st.markdown("---")
        st.markdown("**수치 요약**")
        diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
        diag_col1.metric("적분 성공 여부", "성공" if sol.success else "실패")
        diag_col2.metric("Y(t) 최댓값", f"{Y_arr.max():.3f}")
        diag_col3.metric("C(t) 최댓값", f"{C_arr.max():.3f}")
        diag_col4.metric("C(t) 최솟값", f"{C_arr.min():.3f}")

    st.markdown("---")
    st.subheader("B. Cox 모형 vs ODE 모형 — 예측이 어긋나는 지점 역추적")

    if "cph_model" not in st.session_state or "ode_solution" not in st.session_state:
        st.info("1번 탭에서 Cox 모형을, 2번 탭에서 ODE 시뮬레이션을 먼저 실행해주세요.")
    else:
        cph = st.session_state["cph_model"]
        sol = st.session_state["ode_solution"]
        p = st.session_state["ode_params"]

        # Cox 모형에서 aerobic_pa, BMI 계수 부호 확인
        cox_summary = cph.summary
        contradictions = []

        if "aerobic_pa" in cox_summary.index:
            hr_aerobic = cox_summary.loc["aerobic_pa", "exp(coef)"]
            p_aerobic = cox_summary.loc["aerobic_pa", "p"]
            # ODE 모형은 운동(M)이 크면 YAP/TAZ가 낮아지고 암세포 증식이 억제되어야 함(위험 감소 예측)
            if hr_aerobic > 1 and p_aerobic < 0.05:
                contradictions.append((
                    "운동(유산소 신체활동) 변수의 방향 불일치",
                    f"Cox 모형: 유산소 운동 실천군의 위험비 HR={hr_aerobic:.2f} (p={p_aerobic:.3f}) "
                    f"→ 통계적으로 **운동군이 오히려 암 진단 위험이 더 높게** 나타남",
                    f"ODE 모형: M(운동에 의한 AMPK 활성)이 클수록 YAP/TAZ 정상상태 "
                    f"Y*=αA/(βM+γ)가 낮아지고, 이에 따라 암세포 증식 가속 항(δYC)이 줄어 "
                    f"**운동이 암 억제 방향으로 작동**한다고 예측함",
                    "가능한 원인 (역추적):\n\n"
                    "1. **역인과관계(reverse causation)**: 이 자료는 단면조사이므로, 응답자가 "
                    "'암 진단 이후'에 운동을 시작했거나(관리 목적), 반대로 컨디션이 나빠 "
                    "운동을 못 하게 된 것이 아니라 오히려 진단 후 치료·관찰 과정에서 "
                    "운동을 하게 된 경우가 섞여 있을 수 있습니다. `time`(진단 시 나이)이 "
                    "과거 시점인데 `aerobic_pa`(운동 여부)는 2024년 현재 시점 측정값이라, "
                    "실제로는 시간 순서가 뒤바뀐 상태로 모형에 들어갔을 가능성이 있습니다.\n\n"
                    "2. **혼란변수(confounding)**: 암 생존자들이 재발 방지나 건강 관리 "
                    "목적으로 운동을 더 적극적으로 실천하게 되는 경향(surveillance bias와 "
                    "유사)이 있을 수 있습니다.\n\n"
                    "3. **ODE 모형의 단순화**: ODE 모형은 '운동 → AMPK ↑ → YAP/TAZ ↓ → "
                    "암세포 억제'라는 단일 경로만 가정했지만, 실제로는 운동이 면역계·염증 "
                    "반응 등 다른 경로에도 영향을 미치며, 그 경로들이 이 모형에 포함되지 "
                    "않았습니다."
                ))

        if "BMI" in cox_summary.index:
            hr_bmi = cox_summary.loc["BMI", "exp(coef)"]
            p_bmi = cox_summary.loc["BMI", "p"]
            if p_bmi >= 0.05:
                contradictions.append((
                    "BMI 변수의 유의성 불일치",
                    f"Cox 모형: BMI의 위험비 HR={hr_bmi:.3f} (p={p_bmi:.3f}) "
                    f"→ 통계적으로 **유의하지 않음** (귀무가설 기각 못함)",
                    f"ODE 모형: A(비만에 의한 Akt 활성)가 클수록 YAP/TAZ 정상상태가 "
                    f"선형으로 증가하고(Y*=αA/(βM+γ)), 암세포 증식도 그에 따라 가속된다고 "
                    f"**뚜렷하게 예측함**",
                    "가능한 원인 (역추적):\n\n"
                    "1. **검정력 부족**: 전체 표본 대비 암 진단자가 소수(340명 내외)라 "
                    "BMI처럼 효과크기가 크지 않은 변수는 통계적으로 유의하게 검출되지 "
                    "않을 수 있습니다. ODE 모형은 표본 크기와 무관하게 '메커니즘상 이런 "
                    "효과가 있어야 한다'는 이론적 예측이라, 표본 부족의 영향을 받지 않습니다.\n\n"
                    "2. **BMI ≠ Akt 활성도**: ODE 모형의 A는 실제로는 세포 내 Akt 인산화 "
                    "수준을 뜻하지만, Cox 모형에서는 이를 측정할 수 없어 BMI로 근사했습니다. "
                    "BMI가 높아도 인슐린 감수성이 유지되는 사람(대사적으로 건강한 비만)은 "
                    "Akt 활성화가 크지 않을 수 있어, 대리변수(proxy)로서 BMI의 설명력에 "
                    "한계가 있습니다.\n\n"
                    "3. **비선형 관계 가능성**: Cox 모형은 BMI와 로그위험도 사이에 "
                    "선형관계를 가정하지만, 실제로는 정상체중 구간에서는 영향이 미미하다가 "
                    "고도비만 구간에서만 급격히 커지는 비선형 관계일 수 있습니다."
                ))

        if len(contradictions) == 0:
            st.success("현재 설정에서는 Cox 모형과 ODE 모형의 예측 방향이 뚜렷하게 "
                       "어긋나는 지점이 발견되지 않았습니다.")
        else:
            for title, cox_result, ode_result, root_cause in contradictions:
                with st.expander(f"⚠️ {title}", expanded=True):
                    st.markdown(f"**통계적 관측(Cox)**: {cox_result}")
                    st.markdown(f"**기전적 예측(ODE)**: {ode_result}")
                    st.markdown(f"**역추적한 원인 후보**\n\n{root_cause}")

# ==================================================================
# 탭 4. 민감도 분석
# ==================================================================
with tab4:
    st.header("민감도 분석: 가정치 파라미터의 한계 보완")

    st.markdown("""
    2번 탭의 α, β, γ, r, K, δ는 문헌에서 직접 추정한 값이 아니라 **가정치**입니다.
    가정치 하나로 얻은 결과만 보고하면 "왜 하필 그 숫자를 썼는가"라는 질문에 답하기
    어렵습니다. 대신 여기서는 **파라미터를 체계적으로 바꿔가며 결과가 얼마나 민감하게
    달라지는지**를 보여줌으로써, 특정 수치 자체보다 **정성적 패턴**(운동이 늘수록
    암세포 억제 효과가 있는가, 그 관계가 선형적인가 등)이 얼마나 안정적으로 유지되는지
    검증합니다.
    """)

    if "ode_params" not in st.session_state:
        st.info("먼저 2번 탭에서 미분방정식을 한 번 실행해 기준 파라미터를 설정해주세요.")
    else:
        base_p = st.session_state["ode_params"]

        st.subheader("파라미터 하나를 바꿀 때 최종 암세포 수는 어떻게 변하는가")

        sens_param = st.selectbox(
            "민감도를 분석할 파라미터 선택",
            options=["alpha (Akt→YAP/TAZ 생성)", "beta (AMPK→YAP/TAZ 분해)",
                     "delta (YAP/TAZ→암세포 가속)", "M (운동 강도)", "A (비만 강도)"],
            help="선택한 파라미터만 아래 범위에서 변화시키고, 나머지는 2번 탭에서 "
                 "설정한 값으로 고정합니다."
        )
        param_key_map = {
            "alpha (Akt→YAP/TAZ 생성)": "alpha",
            "beta (AMPK→YAP/TAZ 분해)": "beta",
            "delta (YAP/TAZ→암세포 가속)": "delta",
            "M (운동 강도)": "M",
            "A (비만 강도)": "A",
        }
        key = param_key_map[sens_param]

        default_ranges = {
            "alpha": (0.0, 2.0),
            "beta": (0.0, 2.0),
            "delta": (0.0, 0.1),
            "M": (0.0, 5.0),
            "A": (0.0, 5.0),
        }
        lo_default, hi_default = default_ranges[key]
        range_vals = st.slider(
            f"{key} 값 범위", lo_default, hi_default,
            (lo_default, hi_default), step=(hi_default - lo_default) / 100
        )
        n_points = st.slider("분석할 지점 수", 5, 50, 20)

        sweep_values = np.linspace(range_vals[0], range_vals[1], n_points)
        final_C = []
        final_Y = []

        for v in sweep_values:
            local_p = dict(base_p)
            local_p[key] = v
            sol_local = solve_ivp(
                yap_taz_model, (0, base_p["t_max"]), [base_p["Y0"], base_p["C0"]],
                args=(local_p["alpha"], local_p["beta"], local_p["gamma"], local_p["A"],
                      local_p["r"], local_p["K"], local_p["delta"], local_p["M"]),
                t_eval=np.linspace(0, base_p["t_max"], 200), method="RK45",
            )
            final_C.append(sol_local.y[1][-1])
            final_Y.append(sol_local.y[0][-1])

        fig4 = make_subplots(rows=1, cols=2,
                              subplot_titles=(f"{key} 변화에 따른 최종 YAP/TAZ 농도",
                                               f"{key} 변화에 따른 최종 암세포 수"))
        fig4.add_trace(go.Scatter(x=sweep_values, y=final_Y, mode="lines+markers",
                                    line=dict(color="#DD8452")), row=1, col=1)
        fig4.add_trace(go.Scatter(x=sweep_values, y=final_C, mode="lines+markers",
                                    line=dict(color="#C44E52")), row=1, col=2)
        fig4.update_xaxes(title_text=key, row=1, col=1)
        fig4.update_xaxes(title_text=key, row=1, col=2)
        fig4.update_yaxes(title_text="Y(t_max)", row=1, col=1)
        fig4.update_yaxes(title_text="C(t_max)", row=1, col=2)
        fig4.update_layout(template="plotly_white", height=420, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

        # 민감도 지표: (결과 변화율) / (파라미터 변화율)
        pct_change_param = (range_vals[1] - range_vals[0]) / (abs(range_vals[0]) + abs(range_vals[1]) + 1e-9)
        pct_change_C = (max(final_C) - min(final_C)) / (abs(min(final_C)) + 1e-9)
        st.markdown(f"""
        **민감도 요약**: `{key}`를 {range_vals[0]:.3f} → {range_vals[1]:.3f}로 바꾸는 동안,
        최종 암세포 수는 {min(final_C):.2f} → {max(final_C):.2f}로 변했습니다
        (상대 변화율 약 {pct_change_C*100:.1f}%).

        이 비율이 클수록 모형의 결론이 해당 파라미터의 정확한 값에 크게 의존한다는 뜻이므로,
        보고서에서는 "이 파라미터는 가정치이며, 그 값에 따라 결과가 최대 {pct_change_C*100:.0f}%까지
        달라질 수 있다"는 식으로 한계를 명시하는 것이 좋습니다.
        """)

        st.subheader("2차원 민감도: 비만(A)과 운동(M)의 상대적 힘겨루기")
        st.caption("Akt 신호(A, 비만)와 AMPK 신호(M, 운동)를 동시에 바꿔가며 "
                   "최종 암세포 수를 히트맵으로 표현합니다.")

        A_range = np.linspace(0.1, 5.0, 15)
        M_range = np.linspace(0.1, 5.0, 15)
        heatmap_C = np.zeros((len(M_range), len(A_range)))

        for i, m_val in enumerate(M_range):
            for j, a_val in enumerate(A_range):
                sol_hm = solve_ivp(
                    yap_taz_model, (0, base_p["t_max"]), [base_p["Y0"], base_p["C0"]],
                    args=(base_p["alpha"], base_p["beta"], base_p["gamma"], a_val,
                          base_p["r"], base_p["K"], base_p["delta"], m_val),
                    t_eval=[base_p["t_max"]], method="RK45",
                )
                heatmap_C[i, j] = sol_hm.y[1][-1]

        fig5 = go.Figure(data=go.Heatmap(
            z=heatmap_C, x=A_range, y=M_range,
            colorscale="RdYlGn_r",
            colorbar=dict(title="최종 암세포 수"),
        ))
        fig5.update_layout(
            title="비만 신호(A) × 운동 신호(M)에 따른 최종 암세포 수",
            xaxis_title="A: 비만/고혈당에 의한 Akt 활성",
            yaxis_title="M: 운동에 의한 AMPK 활성",
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("""
        히트맵에서 **오른쪽 아래로 갈수록(A 높음, M 낮음)** 암세포 수가 많아지는 패턴이
        일관되게 나타난다면, 이는 "비만 신호가 강하고 운동 신호가 약할 때 암세포 증식이
        가속된다"는 모형의 정성적 결론이 파라미터 선택에 크게 좌우되지 않고 **구조적으로
        안정적**이라는 근거가 됩니다. 반대로 이 패턴이 파라미터 조합에 따라 자주 뒤집힌다면,
        결론을 더 조심스럽게 제시해야 합니다.
        """)

st.markdown("---")
st.caption("""
**분석 자료**: 질병관리청 국민건강영양조사 제9기 2차년도(2024, HN24) 원시자료 |
**방법론 한계**: 단면조사 자료 특성상 인과관계를 확정할 수 없으며, YAP/TAZ 미분방정식
모형의 파라미터는 문헌 추정치가 아닌 가정치임을 명시함
""")
