# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH01-Basic")

question = '대한민국의 수도는 어디인가요?'

## 2. llm log Prob - 주어진 텍스트에 대한 모델의 토큰 확률의 로그값을 보여준다
# 모델이 해당 토큰을 예측할 확률값을 보여준다.

llm_with_logprob = ChatOpenAI(
    temperature= 0.1,
    max_tokens=2048,
    model_name='gpt-3.5-turbo'
).bind(logprobs=True)


response = llm_with_logprob.invoke(question)
print(response.response_metadata)


## 3. 