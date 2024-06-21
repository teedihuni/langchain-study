from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH04-Model')

from langchain_openai import ChatOpenAI
from langchain_community.llms.loading import load_llm
from langchain.prompts import PromptTemplate

print(f"ChatOpenAI: {ChatOpenAI.is_lc_serializable()}")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate.from_template('{fruit}의 색상이 무엇입니까?')
print(prompt)

chain = prompt | llm
print(chain.is_lc_serializable())


## chain 자체를 저장하고 불러올 수 있다.
from langchain.load import dumpd
import pickle

dumped_chain = dumpd(chain)
with open('fruit_chain.pkl','wb') as f:
    pickle.dump(dumped_chain, f)

with open("fruit_chain.pkl", "rb") as f:
    loaded_chain = pickle.load(f)

print(loaded_chain)

# 불러오기
from langchain.load.load import load

loaded_chain = load(loaded_chain)
print(loaded_chain.invoke({"fruit":"사과"}))