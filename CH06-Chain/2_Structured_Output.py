from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

'''
## create_structured_output_runnable
구조화된 chain을 형성.
특정 주제에 대한 4지선다형 퀴즈를 생성하는 과정을 구현.
Quiz 클래스는 퀴즈의 질문, 난이도, 그리고 네 개의 선택지를 정의.

'''

class Quiz(BaseModel):
    """4지선다형 퀴즈의 정보를 추출합니다"""

    question: str = Field(..., description="퀴즈의 질문")
    level: str = Field(
        ..., description="퀴즈의 난이도를 나타냅니다. (쉬움, 보통, 어려움)"
    )
    options: List[str] = Field(..., description="퀴즈의 4개의 선택지 입니다.")

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-famous quizzer and generates quizzes in structured formats.",
        ),
        (
            "human",
            "TOPIC 에 제시된 내용과 관련한 4지선다형 퀴즈를 출제해 주세요. 만약, 실제 출제된 기출문제가 있다면 비슷한 문제를 만들어 출제하세요."
            "단, 문제에 TOPIC 에 대한 내용이나 정보는 포함하지 마세요. \nTOPIC:\n{topic}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = create_structured_output_runnable(Quiz, llm, prompt)
generated_quiz = chain.invoke({"topic": "ADSP(데이터 분석 준전문가) 자격 시험"})

print(generated_quiz)

# 문제풀기 좋은 형태로 출력하기
print(f"{generated_quiz.question} (난이도: {generated_quiz.level})\n")
for i, opt in enumerate(generated_quiz.options):
    print(f"{i+1}) {opt}")