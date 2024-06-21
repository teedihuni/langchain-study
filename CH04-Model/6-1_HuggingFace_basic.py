from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_teddynote import logging
import os

load_dotenv()
logging.langsmith('CH04-Model')
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
## 주의사항
# - 허깅페이스 token 생성시에 api호출해서 model 사용할 것인지에 대해 설정해줘야함
# - 그냥 바로 발급하면 어떠한 기능도 check 되어있지 않아 로그인은 되지만 유효한 토큰이 아니라는 에러 메세지를 겪을것임.
#login()

from langchain.prompts import PromptTemplate

template = """Please answer the following questions concisely.
QUESTION: {question}

ANSWER: """

prompt = PromptTemplate.from_template(template)

from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#repo_id = 'Qwen/Qwen2-7B-Instruct-GGUF'

llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # 모델 저장소 ID를 지정합니다.
    max_new_tokens=256,  # 생성할 최대 토큰 길이를 설정합니다.
    temperature=0.1,  # 샘플링 온도를 설정합니다. 값이 높을수록 더 다양한 출력을 생성합니다.
    callbacks=[StreamingStdOutCallbackHandler()],  # 콜백을 설정합니다.
    streaming=True,  # 스트리밍을 사용합니다.
)

# LLMChain을 초기화하고 프롬프트와 언어 모델을 전달합니다.
llm_chain = prompt | llm
# 질문을 전달하여 LLMChain을 실행하고 결과를 출력합니다.
response = llm_chain.invoke(
    {"question": "Please tell me top 5 places to visit in Seoul, Korea."}
)
print(response)

## Endpoint 설정

# Inference Endpoint URL을 아래에 설정합니다.
your_endpoint_url = "https://api-inference.huggingface.co/models/google/gemma-7b"

llm = HuggingFaceEndpoint(
    # 엔드포인트 URL을 설정합니다.
    endpoint_url=f"{your_endpoint_url}",
    # 생성할 최대 토큰 수를 설정합니다.
    max_new_tokens=512,
    # 상위 K개의 토큰을 선택합니다.
    top_k=10,
    # 누적 확률이 top_p에 도달할 때까지 토큰을 선택합니다.
    top_p=0.95,
    # typical_p 확률 이상의 토큰만 선택합니다.
    typical_p=0.95,
    # 샘플링 온도를 설정합니다. 낮을수록 더 결정적입니다.
    temperature=0.01,
    # 반복 패널티를 설정합니다. 높을수록 반복을 줄입니다.
    repetition_penalty=1.03,
)
# 주어진 프롬프트에 대해 언어 모델을 실행합니다.
result = llm.invoke(input="#QUESTION: 대한민국의 수도는 어디인가요?\n\n#ANSWER:")
print(result)