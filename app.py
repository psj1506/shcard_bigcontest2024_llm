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
GOOGLE_API_KEY = st.secrets["import os
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

# 사이드바 설정
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

    price_options = ['👌 상관 없음','😎 최고가', '💸 고가', '💰 평균 가격대', '💵 중저가', '😂 저가']
    price_mapping = {
        '👌 상관 없음': '상관 없음',
        '😎 최고가': '최고가',
        '💸 고가': '고가',
        '💰 평균 가격대': '평균 가격대',
        '💵 중저가': '중저가',
        '😂 저가': '저가'
    }
    selected_price = st.sidebar.selectbox("", price_options, key="price")
    price = price_mapping.get(selected_price, '상관 없음')

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index_1.index')):
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
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file_1.npy'))
embeddings_tour = np.load(os.path.join(module_path, 'embeddings_tour_array_file_1.npy'))

# 질문 파싱 함수
def parse_question(question):
    """
    질문을 파싱하여 필터링 기준을 추출합니다.
    현재는 위치와 연령대를 추출합니다.
    필요에 따라 추가적인 필터링 기준을 추출하도록 확장할 수 있습니다.
    """
    location_match = re.search(r'제주시 한림읍', question)
    age_group_match = re.search(r'(\d+)대', question)
    
    location = location_match.group() if location_match else None
    age_group = age_group_match.group(1) if age_group_match else None
    
    return location, age_group

# FAISS를 활용한 응답 생성
def generate_response_with_faiss(question, df, embeddings, model, df_tour, embeddings_tour, k=3, print_prompt=True):
    location, age_group = parse_question(question)
    
    if not location:
        return "질문에서 위치 정보를 찾을 수 없습니다. 다시 입력해주세요."
    
    # 위치에 따라 데이터 필터링
    filtered_df = df[df['가맹점주소'].str.contains(location)].copy()
    
    # 가격대 필터링 로직 수정
    if price != '상관 없음':
        price_filter = {
            '최고가': '6',
            '고가': '5',
            '평균 가격대': ('3', '4'),
            '중저가': '2',
            '저가': '1'
        }
        if price in price_filter:
            if isinstance(price_filter[price], tuple):
                # startswith expects a single string or tuple of strings
                filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price_filter[price])].reset_index(drop=True)
            else:
                filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price_filter[price])].reset_index(drop=True)
    
    # FAISS 인덱스 재구성 (필터링된 데이터로)
    if len(filtered_df) == 0:
        return "질문과 일치하는 가게가 없습니다."
    
    # 텍스트 컬럼이 있는지 확인
    if 'text' not in filtered_df.columns:
        return "데이터셋에 'text' 컬럼이 없습니다."
    
    # 임베딩 생성 (필터링된 데이터)
    filtered_embeddings = np.array([embed_text(text) for text in filtered_df['text']])
    
    # FAISS 인덱스 생성
    dimension = filtered_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(filtered_embeddings)
    
    # FAISS 검색
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k)
    
    # 검색 결과가 없는 경우
    if indices.size == 0:
        return "질문과 일치하는 가게가 없습니다."
    
    # 검색된 카페들 선택
    top_cafes = filtered_df.iloc[indices[0, :]].copy()
    
    # 가장 높은 30대 이용 비중을 가진 카페 선택
    if not top_cafes.empty:
        top_cafe = top_cafes.loc[top_cafes['최근12개월30대회원수비중'].idxmax()]
        reference_info = f"{top_cafe['가맹점명']} - {top_cafe['가맹점주소']} - 30대 비중: {top_cafe['최근12개월30대회원수비중'] * 100:.1f}%"
    else:
        reference_info = "질문과 일치하는 가게가 없습니다."
    
    # 관광지 정보 필터링 (필요 시 수정 가능)
    reference_tour = "\n".join(df_tour['text'].iloc[:1])  # 예시: 첫 번째 관광지 정보
    
    prompt = f"""질문: {question}
대답시 필요한 내용: 
- 근처 음식점을 추천할때는 질문에 주소에 대한 정보가 있다면 음식점의 주소가 비슷한지 확인해.
- 차로 이동시간이 얼마인지 알려줘. 추천해줄때 이동시간을 고려해서 답변해줘.
- 가맹점업종이 커피인 가게는 업종이 카페야.
- 대답해줄때 업종별로 가능하면 하나씩 추천해줘.
- 그리고 추가적으로 그 중에서 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.
참고할 정보: {reference_info}
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

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = generate_response_with_faiss(prompt, df, embeddings, model, df_tour, embeddings_tour)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 로그 기록
        logging.info(f"Question: {prompt}")
        logging.info(f"Answer: {response}")

# 피드백 폼 추가
def get_feedback():
    st.sidebar.header("💬 피드백")
    feedback = st.sidebar.text_area("추천에 대한 피드백을 남겨주세요.")
    if st.sidebar.button("제출"):
        if feedback:
            with open("feedback.log", "a", encoding="utf-8") as f:
                f.write(f"Feedback: {feedback}\n")
            st.sidebar.success("피드백이 제출되었습니다. 감사합니다!")
        else:
            st.sidebar.warning("피드백을 입력해주세요.")

# 피드백 폼 호출
get_feedback()
"]
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

# 사이드바 설정
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

    price_options = ['👌 상관 없음','😎 최고가', '💸 고가', '💰 평균 가격대', '💵 중저가', '😂 저가']
    price_mapping = {
        '👌 상관 없음': '상관 없음',
        '😎 최고가': '최고가',
        '💸 고가': '고가',
        '💰 평균 가격대': '평균 가격대',
        '💵 중저가': '중저가',
        '😂 저가': '저가'
    }
    selected_price = st.sidebar.selectbox("", price_options, key="price")
    price = price_mapping.get(selected_price, '상관 없음')

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index_1.index')):
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
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file_1.npy'))
embeddings_tour = np.load(os.path.join(module_path, 'embeddings_tour_array_file_1.npy'))

# 질문 파싱 함수
def parse_question(question):
    """
    질문을 파싱하여 필터링 기준을 추출합니다.
    현재는 위치와 연령대를 추출합니다.
    필요에 따라 추가적인 필터링 기준을 추출하도록 확장할 수 있습니다.
    """
    location_match = re.search(r'제주시 한림읍', question)
    age_group_match = re.search(r'(\d+)대', question)
    
    location = location_match.group() if location_match else None
    age_group = age_group_match.group(1) if age_group_match else None
    
    return location, age_group

# FAISS를 활용한 응답 생성
def generate_response_with_faiss(question, df, embeddings, model, df_tour, embeddings_tour, k=3, print_prompt=True):
    location, age_group = parse_question(question)
    
    if not location:
        return "질문에서 위치 정보를 찾을 수 없습니다. 다시 입력해주세요."
    
    # 위치에 따라 데이터 필터링
    filtered_df = df[df['가맹점주소'].str.contains(location)].copy()
    
    # 가격대 필터링 로직 수정
    if price != '상관 없음':
        price_filter = {
            '최고가': '6',
            '고가': '5',
            '평균 가격대': ('3', '4'),
            '중저가': '2',
            '저가': '1'
        }
        if price in price_filter:
            if isinstance(price_filter[price], tuple):
                # startswith expects a single string or tuple of strings
                filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price_filter[price])].reset_index(drop=True)
            else:
                filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith(price_filter[price])].reset_index(drop=True)
    
    # FAISS 인덱스 재구성 (필터링된 데이터로)
    if len(filtered_df) == 0:
        return "질문과 일치하는 가게가 없습니다."
    
    # 텍스트 컬럼이 있는지 확인
    if 'text' not in filtered_df.columns:
        return "데이터셋에 'text' 컬럼이 없습니다."
    
    # 임베딩 생성 (필터링된 데이터)
    filtered_embeddings = np.array([embed_text(text) for text in filtered_df['text']])
    
    # FAISS 인덱스 생성
    dimension = filtered_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(filtered_embeddings)
    
    # FAISS 검색
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k)
    
    # 검색 결과가 없는 경우
    if indices.size == 0:
        return "질문과 일치하는 가게가 없습니다."
    
    # 검색된 카페들 선택
    top_cafes = filtered_df.iloc[indices[0, :]].copy()
    
    # 가장 높은 30대 이용 비중을 가진 카페 선택
    if not top_cafes.empty:
        top_cafe = top_cafes.loc[top_cafes['최근12개월30대회원수비중'].idxmax()]
        reference_info = f"{top_cafe['가맹점명']} - {top_cafe['가맹점주소']} - 30대 비중: {top_cafe['최근12개월30대회원수비중'] * 100:.1f}%"
    else:
        reference_info = "질문과 일치하는 가게가 없습니다."
    
    # 관광지 정보 필터링 (필요 시 수정 가능)
    reference_tour = "\n".join(df_tour['text'].iloc[:1])  # 예시: 첫 번째 관광지 정보
    
    prompt = f"""질문: {question}
대답시 필요한 내용: 
- 근처 음식점을 추천할때는 질문에 주소에 대한 정보가 있다면 음식점의 주소가 비슷한지 확인해.
- 차로 이동시간이 얼마인지 알려줘. 추천해줄때 이동시간을 고려해서 답변해줘.
- 가맹점업종이 커피인 가게는 업종이 카페야.
- 대답해줄때 업종별로 가능하면 하나씩 추천해줘.
- 그리고 추가적으로 그 중에서 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.
참고할 정보: {reference_info}
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

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = generate_response_with_faiss(prompt, df, embeddings, model, df_tour, embeddings_tour)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 로그 기록
        logging.info(f"Question: {prompt}")
        logging.info(f"Answer: {response}")

# 피드백 폼 추가
def get_feedback():
    st.sidebar.header("💬 피드백")
    feedback = st.sidebar.text_area("추천에 대한 피드백을 남겨주세요.")
    if st.sidebar.button("제출"):
        if feedback:
            with open("feedback.log", "a", encoding="utf-8") as f:
                f.write(f"Feedback: {feedback}\n")
            st.sidebar.success("피드백이 제출되었습니다. 감사합니다!")
        else:
            st.sidebar.warning("피드백을 입력해주세요.")

# 피드백 폼 호출
get_feedback()
