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
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 요약문을 작성하기 위한 프롬프트 정의 (직접 프롬프트를 작성하는 경우)
# prompt_template = """Please summarize the sentence according to the following REQUEST.
# REQUEST:
# 1. Summarize the main points in bullet points in KOREAN.
# 2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
# 3. Use various emojis to make the summary more interesting.
# 4. Translate the summary into Korean if it is written in English.
# 5. DO NOT translate any technical terms.
# 6. DO NOT include any unnecessary information.
# CONTEXT:
# {context}

# SUMMARY:"
# """
# prompt = PromptTemplate.from_template(prompt_template)

# 원격 저장소에서 프롬프트를 가져오는 경우
prompt = hub.pull("teddynote/summary-stuff-documents-korean")
print(prompt.template)

# LLM 체인 정의
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-16k",
    streaming=True,
    callbacks=[StreamCallback()],
)

## [기존 코드]
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
# docs = loader.load()
# response = stuff_chain.invoke({"input_documents": docs})

# LLMChain 방식이 deprecated되어 LCEL 방식으로 변경해서 코드 작성
llm_chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

docs = loader.load()
response = llm_chain.invoke({"input_documents": docs})



