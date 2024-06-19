from dotenv import load_dotenv
from langchain_teddynote import logging
import ast

load_dotenv()
logging.langsmith('CH02-Prompt')

from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.messages import stream_response

llm = ChatOpenAI(
    temperature= 0,
    model_name = 'gpt-3.5-turbo')

question = "대한민국의 수도는 어디야?"

#print(llm.invoke(question).content)
with open('prompts/examples.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 기존 형태로 변환
examples = ast.literal_eval(content)

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

print(example_prompt.format(**examples[0]))

# FewShot 프롬프트 템플릿 작성
prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt= example_prompt,
    suffix = 'Question:\n{question}\nAnswer:',
    input_variables=['question'],
)

question = 'Google이 창립된 연도에 Bill Gates의 나이는 몇살인가요?'
final_prompt = prompt.format(question = question)

# 바로 추론
print(final_prompt)
print(llm.invoke(final_prompt).content)

# chain 생성
chain = prompt | llm
print(chain.invoke(question).content)

# streaming 형식으로 추론
# stream_response(chain.stream(question))

