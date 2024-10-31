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
data_path = './data'  # 사용자가 제공한 경로
module_path = './modules'  # 사용자가 제공한 경로
faiss_index_path = os.path.join(module_path, 'faiss_index_1.index') 
faiss_tour_index_path = os.path.join(module_path, 'faiss_tour_index_1.index')  

jeju_data_path = os.path.join(data_path, "JEJU_DATA.csv")
jeju_tour_path = os.path.join(data_path, "JEJU_TOUR.csv")

# 로그 설정
logging.basicConfig(filename='chatbot_logs.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gemini 모델 설정 (보안 강화)
GOOGLE_API_KEY = st.secrets["API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 데이터 로드 및 필터링
try:
    df = pd.read_csv(jeju_data_path, encoding='cp949')
    df_tour = pd.read_csv(jeju_tour_path, encoding='cp949')
    text_tour = df_tour['text'].tolist()
    logging.info("데이터 로드 완료.")
    st.write("데이터 로드 완료.")
except FileNotFoundError as e:
    logging.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.stop()
except Exception as e:
    logging.error(f"데이터 로드 중 오류 발생: {e}")
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# 최신연월 데이터만 사용
try:
    df = df.loc[df.groupby('가맹점명')['기준연월'].idxmax()].reset_index(drop=True)
    logging.info("최신연월 데이터 필터링 완료.")
except Exception as e:
    logging.error(f"최신연월 데이터 필터링 실패: {e}")
    st.error(f"최신연월 데이터 필터링 중 오류가 발생했습니다: {e}")
    st.stop()

# 'text' 컬럼 존재 확인
if 'text' not in df.columns:
    logging.error("데이터셋에 'text' 컬럼이 없습니다.")
    st.error("데이터셋에 'text' 컬럼이 없습니다.")
    st.stop()

# 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name).to(device)
    logging.info("임베딩 모델 및 토크나이저 로드 완료.")
except Exception as e:
    logging.error(f"임베딩 모델 로드 실패: {e}")
    st.error("임베딩 모델 로드 중 오류가 발생했습니다.")
    st.stop()

# FAISS 인덱스 로드 함수 정의
def load_faiss_index(index_path):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            logging.info(f"FAISS 인덱스 로드 성공: {index_path}")
            return index
        except Exception as e:
            logging.error(f"FAISS 인덱스 로드 실패: {e}")
            st.error(f"FAISS 인덱스 로드 실패: {e}")
            return None
    else:
        logging.error(f"인덱스 파일을 찾을 수 없습니다: {index_path}")
        st.error(f"인덱스 파일을 찾을 수 없습니다: {index_path}")
        return None

# FAISS 인덱스 로드
faiss_index = load_faiss_index(faiss_index_path)
faiss_tour_index = load_faiss_index(faiss_tour_index_path)  # 관광지 인덱스 로드

if faiss_index is not None:
    logging.info("FAISS 인덱스 로드 완료.")
    st.write("FAISS 인덱스 로드 완료.")
else:
    st.error("FAISS 인덱스 로드에 실패했습니다.")
    st.stop()

if faiss_tour_index is not None:
    logging.info("FAISS 관광지 인덱스 로드 완료.")
    st.write("FAISS 관광지 인덱스 로드 완료.")
else:
    st.error("FAISS 관광지 인덱스 로드에 실패했습니다.")
    st.stop()

# FAISS 인덱스와 임베딩 차원 확인
faiss_dim = faiss_index.d
st.write(f"FAISS 인덱스 차원: {faiss_dim}")
logging.info(f"FAISS 인덱스 차원: {faiss_dim}")

# 텍스트 임베딩 생성 함수 정의
def embed_text(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
            # clean_up_tokenization_spaces=True  # 이 줄을 제거했습니다.
        ).to(device)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy()
    except Exception as e:
        logging.error(f"임베딩 생성 실패: {e}")
        st.error(f"임베딩 생성 실패: {e}")  # 디버깅용
        return None

# 질문 파싱 함수 정의
def parse_question(question):
    """
    질문을 파싱하여 필터링 기준을 추출합니다.
    현재는 위치, 연령대, 음식 종류, 가격대를 추출합니다.
    필요에 따라 추가적인 필터링 기준을 추출하도록 확장할 수 있습니다.
    """
    # 위치 패턴: 다양한 위치 패턴을 인식
    location_pattern = r'([가-힣]+시 [가-힣]+읍|[가-힣]+구 [가-힣]+동)'
    location_match = re.search(location_pattern, question)
    location = location_match.group() if location_match else None
    
    # 연령대 추출: 모든 연령대 추출
    age_groups = re.findall(r'(\d+)대', question)
    age_group = age_groups[0] if age_groups else None  # 첫 번째 연령대만 사용
    
    # 음식 종류 추출: 예시로 '커피', '디저트', '스테이크' 등
    food_type_match = re.search(r'(커피|디저트|스테이크|한식|중식|양식|일식)', question)
    food_type = food_type_match.group() if food_type_match else None
    
    # 가격대 추출: 이미 사이드바에서 선택된 가격대 사용
    # 여기서는 기본적으로 '상관 없음'으로 설정
    price = '상관 없음'
    
    return location, age_group, food_type, price

# Streamlit App UI 설정
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
    selected_price = st.sidebar.selectbox("", price_options, key="price", label_visibility="collapsed")  # label_visibility 수정
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

# FAISS를 활용한 응답 생성 함수 정의
def generate_response_with_faiss(question, df, faiss_index, model, text_tour, k=10, print_prompt=True):
    location, age_group, food_type, price = parse_question(question)
    
    # 필터링 기준이 없을 경우 기본값 설정 또는 사용자에게 추가 정보 요청
    if not location and not food_type and not age_group:
        return "검색에 필요한 정보를 찾지 못했습니다. 위치, 연령대, 음식 종류 등을 포함하여 다시 입력해주세요."
    
    # 임베딩 생성
    query_embedding = embed_text(question)
    if query_embedding is None:
        return "질문에 대한 임베딩을 생성할 수 없습니다."
    
    # 임베딩 차원 확인
    embedding_dim = query_embedding.shape[0]
    if embedding_dim != faiss_index.d:
        logging.error(f"임베딩 차원({embedding_dim})과 FAISS 인덱스 차원({faiss_index.d})이 일치하지 않습니다.")
        st.error(f"임베딩 차원({embedding_dim})과 FAISS 인덱스 차원({faiss_index.d})이 일치하지 않습니다.")
        return "임베딩 차원이 FAISS 인덱스와 일치하지 않습니다."
    
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    logging.info(f"Query Embedding Shape: {query_embedding.shape}, Type: {query_embedding.dtype}")
    
    # FAISS 검색
    try:
        distances, indices = faiss_index.search(query_embedding, k)
        logging.info(f"FAISS 검색 완료: {k}개 결과")
        logging.info(f"Distances: {distances}")
        logging.info(f"Indices: {indices}")
    except Exception as e:  # 예외 처리 수정
        logging.error(f"FAISS 검색 실패: {e}")
        st.error(f"FAISS 검색 중 오류가 발생했습니다: {e}")  # 디버깅용
        return "FAISS 검색 중 오류가 발생했습니다."
    
    # 검색 결과가 없는 경우
    if indices.size == 0 or len(indices[0]) == 0:
        return "질문과 일치하는 가게가 없습니다."
    
    # 검색된 가게들 선택
    try:
        top_cafes = df.iloc[indices[0]].copy()
        logging.info(f"검색된 카페들: {top_cafes['가맹점명'].tolist()}")
    except IndexError as e:
        logging.error(f"인덱스 초과 오류: {e}")
        st.error(f"인덱스 초과 오류: {e}")  # 디버깅용
        return "검색된 결과가 없습니다."
    
    # 추가 필터링
    if location:
        top_cafes = top_cafes[top_cafes['가맹점주소'].str.contains(location)]
    
    if food_type:
        top_cafes = top_cafes[top_cafes['업종'].str.contains(food_type)]
    
    if price and price != '상관 없음':
        top_cafes = top_cafes[top_cafes['건당평균이용금액구간'].str.startswith(price_mapping.get(price, '상관 없음'))]
    
    if age_group:
        # 예시: 30대 이상 비중이 높은 가게로 필터링
        # 실제 기준에 맞게 수정 필요
        top_cafes = top_cafes[top_cafes['최근12개월30대회원수비중'] >= 0.3]
    
    logging.info(f"추가 필터링 후 카페 수: {len(top_cafes)}")
    
    if top_cafes.empty:
        return "질문과 일치하는 가게가 없습니다."
    
    # 가장 높은 30대 이용 비중을 가진 카페 선택
    try:
        top_cafe = top_cafes.loc[top_cafes['최근12개월30대회원수비중'].idxmax()]
        reference_info = f"{top_cafe['가맹점명']} - {top_cafe['가맹점주소']} - 30대 비중: {top_cafe['최근12개월30대회원수비중'] * 100:.1f}%"
        logging.info(f"가장 높은 30대 이용 비중 카페 선택: {top_cafe['가맹점명']}")
    except Exception as e:
        logging.error(f"30대 비중 기준 카페 선택 실패: {e}")
        st.error(f"30대 비중 기준 카페 선택 실패: {e}")  # 디버깅용
        return "가장 적합한 가게를 찾는 중 오류가 발생했습니다."
    
    # 관광지 정보 필터링 (필요 시 수정 가능)
    reference_tour = "\n".join(text_tour[:1])  # 예시: 첫 번째 관광지 정보
    
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
        st.error(f"모델 응답 생성 실패: {e}")  # 디버깅용
        return "모델 응답 생성 중 오류가 발생했습니다."

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = generate_response_with_faiss(prompt, df, faiss_index, model, text_tour, k=10)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 로그 기록
        logging.info(f"Question: {prompt}")
        logging.info(f"Answer: {response}")
