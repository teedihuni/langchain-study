from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH04-Model')

from langchain_openai import ChatOpenAI

# 기본
llm = ChatOpenAI(

    temperature = 0,
    max_tokens = 2048,
    model_name = 'gpt-3.5-turbo',
)

question = '대한민국의 수도는 뭐야?'
print(llm.invoke(question).content)

# 템플릿 사용
from langchain.prompts import PromptTemplate

template = '{country}의 수도는 뭐야?'

prompt = PromptTemplate.from_template(template=template)
print(prompt)

# LLMChian 객체
chain = prompt | llm

# 체인 실행 run()
result = chain.invoke('일본')
result = chain.invoke({"country":"캐나다"})
print(result)

