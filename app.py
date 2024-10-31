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

# Streamlit App UI

st.set_page_config(page_title="🍊제주도 맛집 추천")

# Replicate Credentials
with st.sidebar:
    st.title("**🍊제주도 맛집 추천**")

    st.write("")
    st.markdown("""
        <style>
        .sidebar-text {
        color: #FFEC9D;
        font-size: 18px;
        font-weight: bold;
        }
     </style>
     """, unsafe_allow_html=True)

    st.sidebar.markdown('<p class="sidebar-text">💵희망 가격대는 어떻게 되시나요??</p>', unsafe_allow_html=True)


    price = st.sidebar.selectbox("", ['👌 상관 없음','😎 최고가', '💸 고가', '💰 평균 가격대', '💵 중저가', '😂 저가'], key="price")

    if price == '👌 상관 없음':
        price = '상관 없음'
    elif price == '😎 최고가':
        price = '최고가'
    elif price == '💸 고가':
        price = '고가'
    elif price == '💰 평균 가격대':
        price = '평균 가격대'
    elif price == '💵 중저가':
        price = '중저가'
    elif price == '😂 저가':
        price = '저가'

    st.markdown(
        """
         <style>
         [data-testid="stSidebar"] {
         background-color: #ff9900;
         }
         </style>
        """, unsafe_allow_html=True)
    st.write("")

st.title("어서 와용!👋")
st.subheader("인기 있는 :orange[제주 맛집]🍽️😍 후회는 없을걸?!")

st.write("")

st.write("#흑돼지 #제철 생선회 #해물라면 #스테이크 #한식 #중식 #양식 #일식 #흑백요리사..🤤")

st.write("")

# 이미지 추가
image_path = "https://pimg.mk.co.kr/news/cms/202409/22/news-p.v1.20240922.a626061476c54127bbe4beb0aa12d050_P1.png"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="70%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

st.write("")

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

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(0).total_memory else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")

# 텍스트 임베딩 생성
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 텍스트 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))
embeddings_tour = np.load(os.path.join(module_path, 'embeddings_tour_array_file.npy'))

# FAISS를 활용한 응답 생성
def generate_response_with_faiss(question, df, embeddings, model, df_tour, embeddings_tour,max_count=10, k=3, print_prompt=True):
    index = load_faiss_index()
    index.nprobe = 10
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, 10)

    index_tour = load_faiss_index(index_path=os.path.join(module_path, 'faiss_tour_index.index'))
    query_embedding_tour = embed_text(question).reshape(1, -1)
    distances_tour, indices_tour = index_tour.search(query_embedding_tour, 1)

    filtered_df = df.iloc[indices[0, :]].reset_index(drop=True)
    filtered_df_tour = df_tour.iloc[indices_tour[0, :]].reset_index(drop=True)

    # 희망 가격대 조건을 만족하는 가게들만 필터링
    if price == '상관 없음':
        filtered_df = filtered_df
    elif price == '최고가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('6')].reset_index(drop=True)
    elif price == '고가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('5')].reset_index(drop=True)
    elif price == '평균 가격대':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('3'or '4')].reset_index(drop=True)
    elif price == '저가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('2')].reset_index(drop=True)
    elif price == '최저가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('1')].reset_index(drop=True)

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
        print('-----------------------------'*3)
        print(prompt)
        print('-----------------------------'*3)

    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else response

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# 사용자가 입력한 질문에 대한 응답 생성
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = generate_response_with_faiss(prompt, df, embeddings, model, df_tour, embeddings_tour)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
