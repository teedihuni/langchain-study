from dotenv import load_dotenv
load_dotenv()

import os
print(f"[API KEY]\n{os.environ['GOOGLE_API_KEY']}")

from langchain_teddynote import logging
logging.langsmith('CH04-Model')

from langchain_google_genai import ChatGoogleGenerativeAI,HarmBlockThreshold,HarmCategory
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(
    
    model="gemini-pro",
    safety_settings={
        # 위험한 콘텐츠에 대한 차단 임계값을 설정합니다.
        # 이 경우 위험한 콘텐츠를 차단하지 않도록 설정되어 있습니다.
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    safety_settings={
        # 위험한 콘텐츠에 대한 차단 임계값을 설정합니다.
        # 이 경우 위험한 콘텐츠를 차단하지 않도록 설정되어 있습니다.
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

results = llm.batch(
    [
        "2+2 의 계산 결과는?",  # 2+2의 결과는 무엇인가요?
        "3+5 의 계산 결과는?",  # 3+5의 결과는 무엇인가요?
    ]
)

for res in results:
    print(res.content)  # 각 결과의 내용을 출력합니다.