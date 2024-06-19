from dotenv import load_dotenv
from langchain_teddynote import logging
import ast

load_dotenv()
logging.langsmith('CH02-Prompt')

from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.messages import stream_response
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

llm = ChatOpenAI(
    temperature= 0,
    model_name = 'gpt-3.5-turbo')

# example 로드
with open('prompts/examples.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 파이썬 형태에 맞게 변환
examples = ast.literal_eval(content)

# Vector DB 생성
chroma = Chroma('example_selector', OpenAIEmbeddings())

# 유사도 비교를 통해 select
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,#선택가능한 예시 목록
    OpenAIEmbeddings(), #의미적 유사성을 측정
    Chroma, #임베딩을 저장하고 유사성을 검색할 db클래스
    k=1, #생성할 예시의 수
)
question = 'Google이 창립된 연도에 Bill Gates의 나이는 몇살인가요?'
example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

# 유사도 기반한 예시 선택
selected_examples = example_selector.select_examples({"question" : question})

print(f'입력과 가장 유사한 예시:\n{question}\n')
for example in selected_examples:
    print(f'question:\n{example["question"]}')
    print(f'answer:\n{example["answer"]}')

# FewShot 템플릿 생성
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

# 답변 생성
question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"
example_selector_prompt = prompt.format(question=question)
print(example_selector_prompt)

chain = prompt | llm
print(chain.invoke(question).content)