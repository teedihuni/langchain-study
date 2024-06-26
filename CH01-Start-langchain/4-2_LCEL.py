# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response  # 스트리밍 출력
from langchain_core.prompts import PromptTemplate

# API KEY 정보로드
load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH01-Basic")

## LangChain Expression Language(LCEL)
# 가장 기본적이고 일반적인 방법으로 Chain을 구성
#  | 를 사용하여 prompt와 model을 연결

# from_template 메소드 사용해서 PromptTemplate객체 생성
prompt = PromptTemplate.from_template("{topic} 에 대해 쉽게 설명해주세요.")


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.1,
)

chain = prompt|model

input = {"topic" : '인공지능 모델의 학습 원리'}

# # 출력 1
# answer = chain.invoke(input)
# print(answer)

# 출력 2
answer = chain.stream(input)
stream_response(answer)