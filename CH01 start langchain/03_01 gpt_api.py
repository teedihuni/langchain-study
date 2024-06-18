# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH01-Basic")

question = '대한민국의 수도는 어디인가요?'

## 1. llm 기본 객체
llm = ChatOpenAI(
    temperature = 0.1, # 0~2 사이에서 선택, 
    model_name = 'gpt-3.5-turbo' # 적용 가능한 모델 리스트 gpt-3.5-turbo / gpt-4-turbo / gpt-4o
)

response = llm.invoke(question)
print(response)
print(response.content)
print(response.response_metadata)
