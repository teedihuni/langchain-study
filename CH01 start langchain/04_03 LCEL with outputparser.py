# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response  # 스트리밍 출력
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random

# API KEY 정보로드
load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH01-Basic")
output_parser = StrOutputParser() # 결과값만 str으로 내보내는 기능


# 출력 3
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:
- 한글 해석:
"""

# 프롬프트 템플릿을 이용하여 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(template)

# ChatOpenAI 챗모델을 초기화합니다.
model = ChatOpenAI(model_name="gpt-4-turbo")

# 문자열 출력 파서를 초기화합니다.
output_parser = StrOutputParser()
# 체인을 구성합니다.
chain = prompt | model | output_parser
# 완성된 Chain을 실행하여 답변을 얻습니다.
# 스트리밍 출력을 위한 요청
question_list = [{"question": "저는 식당에 가서 음식을 주문하고 싶어요"},{"question": "미국에서 피자 주문"}]
input = random.choice(question_list)

print(input)
answer = chain.stream(input)
# # 스트리밍 출력
stream_response(answer)