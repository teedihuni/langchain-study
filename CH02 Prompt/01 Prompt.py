from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH02-Prompt")


llm =ChatOpenAI()

## 방법 1. from_template() 
# 치환될 변수를 {변수}로 묶어서 템플릿을 정의
template = "{country}의 수도는 어디인가요?"

prompt = PromptTemplate.from_template(template)
print(prompt)
