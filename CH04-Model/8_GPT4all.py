# https://github.com/nomic-ai/gpt4all
# 코드, 스토리, 대화를 포함한 방대한 양의 깨끗한 어시스턴트 데이터로 학습된 오픈 소스 챗봇 생태계

# https://gpt4all.io/index.html
# 다양한 모델들이 존재함.


# 해당 코드는 ubuntu 20.04 에서 진행 but
# OSError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found 에러 발생

local_path = (
    "./models-starcoder-newbpe-q4_0.gguf"
)

from langchain.prompts import PromptTemplate
from gpt4all import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

prompt = PromptTemplate(
    input_variables=['product'],
    template= 'Name any five companies which makes `{product}` ?'
)

llm = GPT4All(
    model = local_path,
    callbacks = [StreamingStdOutCallbackHandler()],
    streaming = True,
    verbose = True,
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"product" : 'smart phone'})
print(response)