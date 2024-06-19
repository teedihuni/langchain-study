from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-Output-Parsers")


## [JsonOutputParser]
# 사용자가 원하는 JSON 스키마를 지정할 수 있게 해주며, 그 스키마에 맞게 LLM에서 데이터를 조회하여 결과를 도출
# 다만 모델의 용량이 충분해야한다

model = ChatOpenAI(temperature=0)

# 데이터 구조 정의
class Topic(BaseModel):
    description: str = Field(description="Concise description about topic")
    hashtags: str = Field(description="Some keywords in hashtag format")

# 질의 작성
query = "온난화에 대해 알려주세요"

# 파서 설정 및 프롬프트 템플릿에 지시사항 주입
parser = JsonOutputParser(pydantic_object=Topic)
prompt = PromptTemplate(
    template = "Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=['query'],
    partial_variables={
        "format_instructions" : parser.get_format_instructions(),
    }
)

chain = prompt | model | parser
print(chain.invoke({'query':query}))