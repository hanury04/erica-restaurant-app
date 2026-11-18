import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ERICA 학교 앞 식당 추천기",
    layout="wide"
)

CATEGORY_MAPPING = {
    '한식': '한식', '닭갈비': '한식', '삼계탕': '한식', '고기요리': '한식',
    '순대, 순댓국': '한식', '순댓국': '한식', '찌개, 전골': '한식',
    '냉면': '한식', '칼국수': '한식', '감자탕': '한식', '치킨, 닭강정': '한식',
    '양식': '양식', '이탈리아음식': '양식', '피자': '양식',
    '샌드위치': '패스트푸드/간편식', '햄버거': '패스트푸드/간편식', '핫도그': '패스트푸드/간편식',
    '분식': '분식', '종합분식': '분식', '떡볶이': '분식',
    '중식': '중식', '마라탕': '중식',
    '일식': '일식', '돈가스': '일식',
    '베트남음식': '세계음식', '아시아음식': '세계음식', '인도음식': '세계음식',
    '멕시코음식': '세계음식', '멕시코, 남미음식': '세계음식'
}

WEIGHT_PRESETS = {
    "균형 추천 (가성비 30% · 거리 25% · 리뷰 25% · 평점 20%)": (0.30, 0.25, 0.25, 0.20),
    "가성비 우선 (가성비 50% · 거리 20% · 리뷰 15% · 평점 15%)": (0.50, 0.20, 0.15, 0.15),
    "거리 우선 (가성비 20% · 거리 50% · 리뷰 15% · 평점 15%)": (0.20, 0.50, 0.15, 0.15),
    "인기·맛집 우선 (가성비 15% · 거리 15% · 리뷰 35% · 평점 35%)": (0.15, 0.15, 0.35, 0.35),
}


@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="cp949")

    df["대분류"] = df["업종"].map(CATEGORY_MAPPING).fillna("기타")
    df["총가격"] = (
        df["대표메뉴 가격(원)"].astype(str).str.replace(",", "", regex=False).astype(int)
    )
    df["1인분가격"] = (
        df["대표메뉴 1인분 가격(원)"].astype(str).str.replace(",", "", regex=False).astype(int)
    )
    df["리뷰수"] = (
        df["리뷰 수"].astype(str).str.replace(",", "", regex=False).astype(int)
    )
    df["2인이상메뉴여부"] = df["총가격"] > df["1인분가격"]
    df["네이버지도_URL"] = df["상호"].apply(
        lambda name: f"https://map.naver.com/p/search/{urllib.parse.quote(f'안산 상록구 {name}')}"
    )
    return df


def recommend_restaurants(
    df: pd.DataFrame,
    budget: int,
    selected_categories: list[str],
    max_walk_min: int,
    solo_only: bool,
    keyword: str,
    weights: tuple[float, float, float, float],
) -> pd.DataFrame:
    filtered = df.copy()

    if selected_categories:
        filtered = filtered[filtered["대분류"].isin(selected_categories)]

    if solo_only:
        filtered = filtered[filtered["총가격"] <= budget]
    else:
        filtered = filtered[filtered["1인분가격"] <= budget]

    filtered = filtered[filtered["도보 시간(분)"] <= max_walk_min]

    if keyword.strip():
        kw = keyword.strip().lower()
        cond_name = filtered["상호"].str.lower().str.contains(kw, na=False)
        cond_menu = filtered["대표 메뉴"].str.lower().str.contains(kw, na=False)
        cond_sub = filtered["업종"].str.lower().str.contains(kw, na=False)
        filtered = filtered[cond_name | cond_menu | cond_sub]

    if filtered.empty:
        return filtered

    filtered = filtered.copy()

    max_dist = df["도보 거리(m)"].max()
    max_log_rev = np.log1p(df["리뷰수"].max())
    global_mean_rating = df["별점"].mean()
    prior_weight = 300

    filtered["가성비점수"] = (1.0 - (filtered["1인분가격"] / budget)).clip(0.0, 1.0)
    filtered["접근성점수"] = (1.0 - (filtered["도보 거리(m)"] / max_dist)).clip(0.0, 1.0)
    filtered["인기도점수"] = (np.log1p(filtered["리뷰수"]) / max_log_rev).clip(0.0, 1.0)

    raw_rating = filtered["별점"].fillna(global_mean_rating)
    bayes_rating = (
        filtered["리뷰수"] * raw_rating + prior_weight * global_mean_rating
    ) / (filtered["리뷰수"] + prior_weight)
    filtered["보정별점"] = bayes_rating.round(2)
    filtered["평점점수"] = (bayes_rating / 5.0).clip(0.0, 1.0)

    w_price, w_dist, w_pop, w_rate = weights
    filtered["추천점수"] = (
        100.0
        * (
            w_price * filtered["가성비점수"]
            + w_dist * filtered["접근성점수"]
            + w_pop * filtered["인기도점수"]
            + w_rate * filtered["평점점수"]
        )
    ).round(1)

    return filtered.sort_values(by=["추천점수", "리뷰수"], ascending=[False, False])


def main():
    st.title("한양대 ERICA 학교 앞 식당 추천 앱")
    st.caption("예산, 선호 카테고리, 도보 시간, 혼밥 여부를 입력하면 가중치 기반 추천 순위를 계산합니다.")

    df = load_data("erica.restaurant.data - 시트1.csv")
    all_categories = sorted(df["대분류"].unique().tolist())

    with st.sidebar:
        st.header("검색 조건 설정")

        budget = st.slider(
            "1인당 최대 예산 (원)",
            min_value=3500,
            max_value=30000,
            value=10000,
            step=500,
        )

        selected_categories = st.multiselect(
            "선호 음식 종류 (대분류)",
            options=all_categories,
            default=all_categories,
        )

        max_walk_min = st.slider(
            "최대 도보 시간 (분)",
            min_value=1,
            max_value=int(df["도보 시간(분)"].max()),
            value=5,
            step=1,
        )

        solo_only = st.checkbox(
            "혼밥 가능 메뉴만 보기 (2인 이상 세트·공유 메뉴 제외)",
            value=False,
        )

        keyword = st.text_input(
            "메뉴·상호 키워드 검색 (선택)",
            placeholder="예: 제육, 마라탕, 버거, 국밥",
        )

        preset_label = st.radio(
            "추천 우선순위 기준",
            options=list(WEIGHT_PRESETS.keys()),
            index=0,
        )
        weights = WEIGHT_PRESETS[preset_label]

    results = recommend_restaurants(
        df=df,
        budget=budget,
        selected_categories=selected_categories,
        max_walk_min=max_walk_min,
        solo_only=solo_only,
        keyword=keyword,
        weights=weights,
    )

    if results.empty:
        st.warning("선택한 조건에 맞는 식당이 없습니다. 예산을 늘리거나 도보 시간·카테고리 범위를 넓혀보세요.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("조건 부합 식당 수", f"{len(results)}곳")
    col2.metric("평균 1인분 가격", f"{int(results['1인분가격'].mean()):,}원")
    col3.metric("평균 도보 시간", f"{results['도보 시간(분)'].mean():.1f}분")
    col4.metric("1위 추천 식당", results.iloc[0]["상호"])

    st.subheader("상위 추천 TOP 3")
    top_n = min(3, len(results))
    top_cols = st.columns(top_n)

    for idx in range(top_n):
        row = results.iloc[idx]
        with top_cols[idx]:
            with st.container(border=True):
                st.markdown(f"### {idx + 1}위. {row['상호']} ({row['추천점수']}점)")
                st.write(f"**분류**: {row['대분류']} ({row['업종']})")
                st.write(f"**대표 메뉴**: {row['대표 메뉴']}")
                if row["2인이상메뉴여부"]:
                    st.write(
                        f"**가격**: 1인분 {row['1인분가격']:,}원 (대표메뉴 주문가 {row['총가격']:,}원)"
                    )
                else:
                    st.write(f"**가격**: {row['1인분가격']:,}원")
                rating_text = (
                    f"{row['별점']:.2f} (보정 {row['보정별점']:.2f})"
                    if pd.notna(row["별점"])
                    else f"미제공 (보정 {row['보정별점']:.2f})"
                )
                st.write(f"**위치**: 도보 {row['도보 시간(분)']}분 ({row['도보 거리(m)']}m) · {row['주소']}")
                st.write(f"**리뷰/별점**: 리뷰 {row['리뷰수']:,}개 · 별점 {rating_text}")
                st.link_button("네이버 지도에서 보기", url=row["네이버지도_URL"])

    st.subheader("전체 추천 순위 목록")
    display_df = results[
        [
            "추천점수",
            "상호",
            "대분류",
            "업종",
            "대표 메뉴",
            "1인분가격",
            "총가격",
            "도보 시간(분)",
            "도보 거리(m)",
            "리뷰수",
            "별점",
            "보정별점",
            "주소",
            "네이버지도_URL",
        ]
    ].rename(
        columns={
            "1인분가격": "1인분 가격(원)",
            "총가격": "대표메뉴 주문가(원)",
            "리뷰수": "리뷰 수",
            "별점": "원본 별점",
            "보정별점": "베이지안 보정별점",
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "네이버지도_URL": st.column_config.LinkColumn(
                "네이버 지도",
                display_text="지도 열기",
            )
        },
    )

    with st.expander("세부 점수 산출 내역 보기 (0~1 정규화 지표)"):
        detail_df = results[
            ["상호", "추천점수", "가성비점수", "접근성점수", "인기도점수", "평점점수"]
        ].copy()
        for col in ["가성비점수", "접근성점수", "인기도점수", "평점점수"]:
            detail_df[col] = (detail_df[col] * 100).round(1)
        st.dataframe(detail_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()