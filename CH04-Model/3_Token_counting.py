from dotenv import load_dotenv
load_dotenv()
from langchain_teddynote import logging
logging.langsmith('CH04-Model')

## 토큰 사용량 확인
# - 현재 openai api에만 구현되어있음

from langchain_community.callbacks.manager import get_openai_callback #기존 튜토리얼의 코드에서 이와같이 변경됨
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo')

with get_openai_callback() as cb:
    resutl = llm.invoke('대한민국의 수도는 어디야?')
    print(cb)

with get_openai_callback() as cb:
    result = llm.invoke("대한민국의 수도는 어디야?")
    result = llm.invoke("대한민국의 수도는 어디야?")
    print(f"총 사용된 토큰수: \t\t{cb.total_tokens}")
    print(f"프롬프트에 사용된 토큰수: \t{cb.prompt_tokens}")
    print(f"답변에 사용된 토큰수: \t{cb.completion_tokens}")
    print(f"호출에 청구된 금액(USD): \t${cb.total_cost}")