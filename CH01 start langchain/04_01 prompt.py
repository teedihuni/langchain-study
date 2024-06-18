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

# template 정의 
template = '{country}의 수도는 어디인가요?'

# from_template 메소드 사용해서 PromptTemplate객체 생성
prompt_template = PromptTemplate.from_template(template)
prompt = prompt_template.format(country='대한민국')
print(prompt)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.1,
)