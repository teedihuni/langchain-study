# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response
from langchain_openai import ChatOpenAI


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH01-Basic")

question = '대한민국의 수도는 어디인가요?'

## streaming
# 이 방식은 전체 결과를 한번에 내보내는게 아니라 한글자씩 다다다 생성하는 느낌으로 출력 결과를 생성해준다. 
llm = ChatOpenAI(
    temperature = 0.1, # 0~2 사이에서 선택, 
    model_name = 'gpt-3.5-turbo' # 적용 가능한 모델 리스트 gpt-3.5-turbo / gpt-4-turbo / gpt-4o
)

answer = llm.stream('대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!')

for token in answer:
    print(token.content, end="", flush=True)
    
stream_response(answer)
