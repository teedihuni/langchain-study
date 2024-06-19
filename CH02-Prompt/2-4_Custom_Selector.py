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
from langchain_teddynote.messages import stream_response

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

# 질문 설정 (instruction만 존재하는 경우)
question = {
    "instruction": "회의록을 작성해 주세요"
}

# 유사한 예를 적절히 찾지 못한다.
print(example_selector.select_examples(question))

## 이 경우에 사용가능한 것이 바로 CustomExampleSelector

from langchain_teddynote.prompts import CustomExampleSelector

#커스텀 예제선택시 생성
custom_selector= CustomExampleSelector(examples, OpenAIEmbeddings())

# 예제 선택 결과 (적절한 것을 찾음)
print(custom_selector.select_examples(question))


## 전체 구조 설정

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}:\n{input}"),
        ("ai", "{answer}"),
    ]
)

custom_fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=custom_selector,  # 커스텀 예제 선택기 사용
    example_prompt=example_prompt,  # 예제 프롬프트 사용
)

custom_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        few_shot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)

# chain
chain = custom_prompt | llm
question = {
    "instruction": "문서를 요약해 주세요",
    "input": "이 문서는 '2023년 글로벌 경제 전망'에 관한 30페이지에 달하는 상세한 보고서입니다. 보고서는 세계 경제의 현재 상태, 주요 국가들의 경제 성장률, 글로벌 무역 동향, 그리고 다가오는 해에 대한 경제 예측을 다룹니다. 이 보고서는 또한 다양한 경제적, 정치적, 환경적 요인들이 세계 경제에 미칠 영향을 분석하고 있습니다.",
}

# 실행 및 결과 출력
_ = stream_response(chain.stream(question))