from dotenv import load_dotenv
load_dotenv()

import os
print(f"[API KEY]\n{os.environ['GOOGLE_API_KEY']}")

from langchain_teddynote import logging
logging.langsmith('CH04-Model')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
llm = ChatGoogleGenerativeAI(model = 'gemini-pro')

## Basic
result = llm.invoke('자연어처리에 대해서 간략히 설명해 줘')
#print(result.content)


## system message setting
model = ChatGoogleGenerativeAI(
    model = 'gemini-pro',
    convert_system_message_to_human='True', #시스템 메세지를 설정할 수 있음
)

result = model.invoke([

    SystemMessage(content='Answer only yes or no.'), #시스템 메세지 설정
    HumanMessage(content='Is apple a fruit?'), #사람 메세지 설정
    
])

#print(result.content)

## streaming & batching
for chunk in llm.stream("Google 의 기업 역사에 대해서 markdown 형식으로 작성해 줘"):
    print(chunk.content, end="", flush=True)  # 각 청크의 내용을 출력합니다.