import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai
import logging
import re

# 경로 설정
data_path = './data'
module_path = './modules'

# 로그 설정
logging.basicConfig(filename='chatbot_logs.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Gemini 모델 설정
genai.configure(api_key="YOUR_API_KEY")  # 실제 API 키 사용
model = genai.GenerativeModel("gemini-1.5-flash")

# 데이터 로드 및 필터링
df = pd.read_csv(os.path.join(data_path, "JEJU_DATA.csv"), encoding='cp949')
df_tour = pd.read_csv(os.path.join(data_path, "JEJU_TOUR.csv"), encoding='cp949')
text_tour = df_tour['text'].tolist()

# 최신연월 데이터만 사용
df = df.loc[df.groupby('가맹점명')['기준연월'].idxmax()].reset_index(drop=True)

# 'text' 컬럼 존재 확인
if 'text' not in df.columns:
    st.error("데이터셋에 'text' 컬럼이 없습니다.")
    st.stop()

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수 정의
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")

# 텍스트 임베딩 생성 함수 정의
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 질문 파싱 함수 정의
def parse_question(question):
    """
    질문을 파싱하여 필터링 기준을 추출합니다.
    현재는 위치, 연령대, 음식 종류, 가격대를 추출합니다.
    필요에 따라 추가적인 필터링 기준을 추출하도록 확장할 수 있습니다.
    """
    location_pattern = r'([가-힣]+시 [가-힣]+읍|[가-힣]+구 [가-힣]+동)'
    location_match = re.search(location_pattern, question)
    location = location_match.group() if location_match else None
    
    age_groups = re.findall(r'(\d+)대', question)
    age_group = age_groups[0] if age_groups else None
    
    food_type_match = re.search(r'(커피|디저트|스테이크|한식|중식|양식|일식)', question)
    food_type = food_type_match.group() if food_type_match else None
    
    price = '상관 없음'
    
    return location, age_group, food_type, price

# Streamlit App UI 설정
st.set_page_config(page_title="🍊제주도 맛집 추천")

# 사이드바 설정
with st.sidebar:
    st.title("**🍊제주도 맛집 추천**")
    st.sidebar.markdown('<p class="sidebar-text">💵희망 가격대는 어떻게 되시나요??</p>', unsafe_allow_html=True)

    price_options = ['👌 상관 없음', '😎 최고가', '💸 고가', '💰 평균 가격대', '💵 중저가', '😂 저가']
    price_mapping = {
        '👌 상관 없음': '상관 없음',
        '😎 최고가': '6',
        '💸 고가': '5',
        '💰 평균 가격대': ('3', '4'),
        '💵 중저가': '2',
        '😂 저가': '1'
    }
    selected_price = st.sidebar.selectbox("", price_options, key="price")
    price = price_mapping.get(selected_price, '상관 없음')

st.title("어서 와용!👋")
st.subheader("인기 있는 :orange[제주 맛집]🍽️😍 후회는 없을걸?!")

# 대화 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당 찾으시나요?? 위치, 업종 등을 알려주시면 최고의 맛집 추천해드릴게요!"}]

# FAISS를 활용한 응답 생성 함수 정의
def generate_response_with_faiss(question, df, model, df_tour, k=10, print_prompt=True):
    location, age_group, food_type, price = parse_question(question)

    # 임베딩 생성
    query_embedding = embed_text(question)
    if query_embedding is None:
        return "질문에 대한 임베딩을 생성할 수 없습니다."
    
    query_embedding = query_embedding.reshape(1, -1).astype('float32')

    # FAISS 검색
    try:
        faiss_index = load_faiss_index()
        distances, indices = faiss_index.search(query_embedding, k)
        logging.info(f"FAISS 검색 완료: {k}개 결과")
    except Exception as e:
        logging.error(f"FAISS 검색 실패: {e}")
        return "FAISS 검색 중 오류가 발생했습니다."
    
    # 검색된 가게들 선택
    try:
        top_cafes = df.iloc[indices[0]].copy()
        logging.info(f"검색된 카페들: {top_cafes['가맹점명'].tolist()}")
    except IndexError as e:
        logging.error(f"인덱스 초과 오류: {e}")
        return "검색된 결과가 없습니다."
    
    # 추가 필터링
    if location:
        top_cafes = top_cafes[top_cafes['가맹점주소'].str.contains(location)]
    
    if food_type:
        top_cafes = top_cafes[top_cafes['업종'].str.contains(food_type)]
    
    if price and price != '상관 없음':
        if isinstance(price, tuple):
            top_cafes = top_cafes[top_cafes['건당평균이용금액구간'].str.startswith(tuple(price))]
        else:
            top_cafes = top_cafes[top_cafes['건당평균이용금액구간'].str.startswith(price)]
    
    if age_group:
        top_cafes = top_cafes[top_cafes['최근12개월30대회원수비중'] >= 0.3]

    if top_cafes.empty:
        return "질문과 일치하는 가게가 없습니다."

    # 가장 높은 30대 이용 비중을 가진 카페 선택
    top_cafe = top_cafes.loc[top_cafes['최근12개월30대회원수비중'].idxmax()]
    reference_info = f"{top_cafe['가맹점명']} - {top_cafe['가맹점주소']} - 30대 비중: {top_cafe['최근12개월30대회원수비중'] * 100:.1f}%"
    
    # 관광지 정보 필터링
    reference_tour = "\n".join(df_tour['text'].iloc[:1])  # 예시: 첫 번째 관광지 정보
    
    prompt = f"""질문: {question}
대답에 필요한 정보: 
- 근처 음식점을 추천할때는 질문에 주소에 대한 정보가 있다면 음식점의 주소가 비슷한지 확인해.
- 차로 이동시간이 얼마인지 알려줘. 추천해줄때 이동시간을 고려해서 답변해줘.
- 가맹점업종이 커피인 가게는 업종이 카페야.
- 대답해줄때 업종별로 가능하면 하나씩 추천해줘.
- 그리고 추가적으로 그 중에서 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.
참고할 정보: {reference_info}
참고할 관광지 정보: {reference_tour}
응답:"""

    if print_prompt:
        st.write('-----------------------------' * 3)
        st.write(prompt)
        st.write('-----------------------------' * 3)

    # 모델 응답 생성
    try:
        response = model.generate_content(prompt)
        logging.info("모델 응답 생성 완료.")
        return response.text if hasattr(response, 'text') else response
    except Exception as e:
        logging.error(f"모델 응답 생성 실패: {e}")
        return "모델 응답 생성 중 오류가 발생했습니다."

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = generate_response_with_faiss(prompt, df, model, df_tour)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
