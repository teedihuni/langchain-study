from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

'''
##create_stuff_documents_chain
문서 목록을 가져와서 모두 프롬프트에 삽입한 다음, 그 프롬프트를 LLM에 전달

해당 체인은 문서가 작고 대부분의 호출에 몇개만 전달되는 태스크에 적합
'''

# 프롬프트 설정방법 1
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert summarizer. Please summarize the following sentence.",
        ),
        (
            "user",
            "Please summarize the sentence according to the following request."
            "\nREQUEST:\n"
            "1. Summarize the main points in bullet points in Korean."
            "2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence."
            "3. Use various emojis to make the summary more interesting."
            "\n\nCONTEXT: {context}\n\nSUMMARY:",
        ),
    ]
)
print(f">> scratch로 설정한 prompt : \n {prompt} \n")

from langchain import hub
# 프롬프트 설정방법 2
prompt = hub.pull("teddynote/summary-stuff-documents-korean")
print(f">> hub에서 받은 prompt : \n {prompt} \n")


## 뉴스기사 로드 & document 생성
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/news.txt")
docs = loader.load()
print(f"문서의 수: {len(docs)}\n")
print("[메타데이터]\n")
print(docs[0].metadata)
print("\n========= [앞부분] 미리보기 =========\n")
print(docs[0].page_content[:500])
print("\n")


## 요약
from langchain.callbacks.base import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler): # 해당 함수에서 print 해줌
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True) 

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    temperature=0.01,
    callbacks=[MyCallbackHandler()],
)
chain = create_stuff_documents_chain(llm, prompt)
answer = chain.invoke({"context": docs}) 

