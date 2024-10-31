import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai
import logging

# 경로 설정
data_path = './data'
module_path = './modules'

# 로그 설정
logging.basicConfig(filename='chatbot_logs.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Gemini 모델 설정
GOOGLE_API_KEY = st.secrets["API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 데이터 로드
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

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index_1.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        logging.info(f"FAISS 인덱스 로드 성공: {index_path}")
        return index
    else:
        logging.error(f"인덱스 파일을 찾을 수 없습니다: {index_path}")
        return None

# 텍스트 임베딩 생성
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

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

# 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 챗 기록 초기화 버튼
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당 찾으시나요?? 위치, 업종 등을 알려주시면 최고의 맛집 추천해드릴게요!"}]
st.sidebar.button('대화 초기화 🔄', on_click=clear_chat_history)

# FAISS를 활용한 응답 생성 함수 정의
def generate_response_with_faiss(question, df, model, df_tour, k=3):
    index = load_faiss_index()
    
    if index is None:
        return "FAISS 인덱스를 로드할 수 없습니다."
    
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    filtered_df = df.iloc[indices[0, :]].reset_index(drop=True)

    # 가격대 필터링
    if price != '상관 없음':
        if isinstance(price, tuple):
            filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price)].reset_index(drop=True)
        else:
            filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price)].reset_index(drop=True)

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    reference_info = "\n".join(filtered_df['text'])
    reference_tour = "\n".join(df_tour['text'].iloc[:1])  # 첫 번째 관광지 정보

    prompt = f"""질문: {question}\n대답시 필요한 내용: 근처 음식점을 추천할때는 질문에 주소에 대한 정보가 있다면 음식점의 주소가 비슷한지 확인해.\n차로 이동시간이 얼마인지 알려줘. 추천해줄때 이동시간을 고려해서 답변해줘.\n가맹점업종이 커피인 가게는 업종이 카페야.\n대답해줄때 업종별로 가능하면 하나씩 추천해줘. 그리고 추가적으로 그 중에서 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.\n참고할 정보: {reference_info}\n참고할 관광지 정보: {reference_tour}\n응답:"""

    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else response

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = generate_response_with_faiss(prompt, df, model, df_tour)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
