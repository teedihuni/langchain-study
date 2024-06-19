from dotenv import load_dotenv
from langchain_teddynote import logging
import ast

load_dotenv()
logging.langsmith('CH02-Prompt')

from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_teddynote.messages import stream_response
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

llm = ChatOpenAI(
    temperature= 0,
    model_name = 'gpt-3.5-turbo')

# example 로드
with open('prompts/examples2.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 파이썬 형태에 맞게 변환
examples = ast.literal_eval(content)

# Vector DB 생성
chroma = Chroma('fewshot_chat', OpenAIEmbeddings())

# 프롬프트 세팅
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human","{instruction}:\n{input}"),
         ("ai","{answer}"),
    ]
)

# 유사도 비교를 통해 select가능한 함수
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,#선택가능한 예시 목록
    OpenAIEmbeddings(), #의미적 유사성을 측정
    Chroma, #임베딩을 저장하고 유사성을 검색할 db클래스
    k=1, #생성할 예시의 수
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
)

# 질문 설정
question = {
    "instruction": "회의록을 작성해 주세요",
    "input": "2023년 12월 26일, ABC 기술 회사의 제품 개발 팀은 새로운 모바일 애플리케이션 프로젝트에 대한 주간 진행 상황 회의를 가졌다. 이 회의에는 프로젝트 매니저인 최현수, 주요 개발자인 황지연, UI/UX 디자이너인 김태영이 참석했다. 회의의 주요 목적은 프로젝트의 현재 진행 상황을 검토하고, 다가오는 마일스톤에 대한 계획을 수립하는 것이었다. 각 팀원은 자신의 작업 영역에 대한 업데이트를 제공했고, 팀은 다음 주까지의 목표를 설정했다.",
}

# 유사한 예제 선택
print(example_selector.select_examples(question))


# 최종 프롬프트 세팅
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        few_shot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)

chain = final_prompt | llm
print(chain.invoke(question).content)
