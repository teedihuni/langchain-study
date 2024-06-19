from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH02-Prompt")


llm =ChatOpenAI()

# 치환될 변수를 {변수}로 묶어서 템플릿을 정의
template = "{country}의 수도는 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables = ['country'],) # 템플릿 문자열에 있는 변수와 비교하여 불일치 하는 경우 예외를 발생
print(prompt)

print(prompt.format(country = '대한민국'))
