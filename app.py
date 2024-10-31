import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai

# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini 모델 설정
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# 데이터 로드
df = pd.read_csv(os.path.join(data_path, "JEJU_DATA.csv"), encoding='cp949')
df_tour = pd.read_csv(os.path.join(data_path, "JEJU_TOUR.csv"), encoding='cp949')
text_tour = df_tour['text'].tolist()

# 최신연월 데이터만 사용 (필터링된 데이터프레임으로 일관되게 사용)
df = df.loc[df.groupby('가맹점명')['기준연월'].idxmax()].reset_index(drop=True)

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

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")

# 텍스트 임베딩 생성
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# FAISS를 활용한 응답 생성
def generate_response_with_faiss(question, df, embeddings, model, df_tour, embeddings_tour, max_count=10, k=3, print_prompt=True):
    index = load_faiss_index()
    index.nprobe = 10
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, 10)

    index_tour = load_faiss_index(index_path=os.path.join(module_path, 'faiss_tour_index.index'))
    query_embedding_tour = embed_text(question).reshape(1, -1)
    distances_tour, indices_tour = index_tour.search(query_embedding_tour, 1)

    filtered_df = df.iloc[indices[0, :]].reset_index(drop=True)
    filtered_df_tour = df_tour.iloc[indices_tour[0, :]].reset_index(drop=True)

    # 가격대 필터링 (튜플을 사용해 특정 가격대 구간 포함)
    if price != '상관 없음':
        if isinstance(price, tuple):
            filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price)].reset_index(drop=True)
        else:
            filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price)].reset_index(drop=True)

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    reference_info = "\n".join(filtered_df['text'])
    reference_tour = "\n".join(filtered_df_tour['text'])

    prompt = f"""질문: {question}
대답에 참고할 지침:
1. 가맹점명과 가맹점업종 정보를 사용하여 질문에 언급된 업종이나 특정 요리를 제공하는 식당만 추천해줘.
2. 사용자가 원하는 위치(주소 포함)를 기준으로 해당 지역 내의 식당만 추천해줘.
3. 차로 이동시간이 얼마인지 알려줘. 추천해줄때 이동시간을 고려해서 답변해줘.
4. 질문에 언급된 업종이나 특정 요리가 없는 경우 업종별로 하나씩 추천해줘.
5. 이용건수 및 연령대, 성별 비중을 사용하여 특정 이용층(예: 20대, 여성)이 많은 가게로 추천합니다. 대답해줄때 업종별로 가능하면 하나씩 추천해줘.
6. 그 외에도 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.
참고할 음식점 정보: {reference_info}
참고할 관광지 정보: {reference_tour}
응답:"""

    if print_prompt:
        print('-----------------------------' * 3)
        print(prompt)
        print('-----------------------------' * 3)

    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else response

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = generate_response_with_faiss(prompt, df, embeddings, model, df_tour, embeddings_tour)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
