from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith('CH06-Chain')

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.callbacks.base import BaseCallbackHandler

# Load some data to summarize
loader = WebBaseLoader("https://www.aitimes.com/news/articleView.html?idxno=131777")
docs = loader.load()
content = docs[0].page_content

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

prompt = hub.pull("teddynote/chain-of-density-korean")
template = prompt.messages[0].prompt.template # 프롬프트에서 템플릿 텍스트만 출력하기 위함
print(template)


# 모델 구축
model = ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        streaming=True,
        callbacks=[StreamCallback()],
    )
# 체인 구축
chain = (
    prompt
    | model
    | StrOutputParser()
)

# 기존 코드에서 겪은 문제
# 1. 템플릿은 TEXT format으로 출력하도록 되어있음
# 2. 기존 코드로 진행시에 invalid json output 오류 발생

# 추론
result = chain.invoke({"question":content})
#print(result)
