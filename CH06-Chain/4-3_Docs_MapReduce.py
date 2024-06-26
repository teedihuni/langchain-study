from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')


## 문서 집합을 가지고 있을때 내용을 요약하는 경우
# 요약기를 구축할때 llm의 context창에 문서를 어떻게 전달할 것인가가 중요
# chain type 설정시 인자들
# 1. Stuff : 단순히 모든 문서를 단일 프롬프트로 "넣는" 방식, 가장 간단한 방식
# 2. Map-reduce : 각 문서를 "map"단계에서 개별적으로 요약하고, "reduce" 단계에서 요약본들을 최종요약본으로 합치는 방식
# 3. Refine : 입력 문서를 순회하며 반복적으로 답변을 업데이트하여 응답을 구성.
#              각 문서에 대해, 모든 비문서 입력, 현재 문서, 그리고 최신 중간 답변을 LLM chain에 전달하여 새로운 답변을 얻음

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# 웹 기반 문서 로더를 초기화
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(f"{token}", end="", flush=True)

##  방법 1 : Stuff 
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain import hub

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamCallback()],
)

# # map-prompt 를 직접 정의하는 경우 다음의 예시를 참고하세요.
# map_template = """The following is a set of documents
# {docs}
# Based on this list of docs, please identify the main themes
# Helpful Answer:"""
# map_prompt = PromptTemplate.from_template(map_template)

# langchain 허브에서 'rlm/map-prompt'를 가져옵니다.
map_prompt = hub.pull("teddynote/map-prompt")
reduce_prompt = hub.pull("teddynote/reduce-prompt-korean")


# input_variables 추출
input_variables = map_prompt.input_variables 
print("\n >> Input Variables: \n", input_variables)

# template 추출
template = map_prompt.template
print("\n >> Template: \n", template)

map_chain = LLMChain(llm=llm, prompt=map_prompt)