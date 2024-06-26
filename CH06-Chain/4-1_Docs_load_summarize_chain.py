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
# 문서를 로드
docs = loader.load()

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(f"{token}", end="", flush=True)
        
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-16k",
    streaming=True,
    callbacks=[StreamCallback()],
)

## load_summarize_chain
# 단순 요약을 위한 chain
# 요약 체인을 로드, 체인 타입을 'stuff'로 지정
# chain_type 에 "map_reduce" or "refine" 을 제공 가능
chain = load_summarize_chain(llm, chain_type="stuff")


# 문서에 대해 요약 체인을 실행
answer = chain.invoke({"input_documents": docs})
print(answer["output_text"])