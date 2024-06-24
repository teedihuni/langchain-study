from langchain_huggingface import HuggingFaceEndpoint
#from langchain_community.llms import huggingface_hub # 기존의 튜토리얼과 다른 부분임
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
logging.langsmith('CH04-Model')
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"
os.environ["HF_HOME"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"

#model_repo = "yanolja/EEVE-Korean-2.8B-v1.0" #2080 8G에 못올림
model_repo = "Qwen/Qwen2-1.5B-Instruct"
#model_repo = "Qwen/Qwen2-7B-Instruct-q3_k_m"

# 모델 부르기
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id=model_repo,  # 사용할 모델의 ID를 지정
    task="text-generation",  # 수행할 작업을 설정
    # 사용할 GPU 디바이스 번호를 지정
    # "auto"로 설정하면 accelerate 라이브러리를 사용
    device=0,
    batch_size = 1,
    # 파이프라인에 전달할 추가 인자를 설정
    pipeline_kwargs={
        "temperature":1,
        "max_new_tokens": 512},
)

## Create Chain

template = """Answer the following question in Korean.
#Question:
{question}

#Answer: """ # 질문 답변 형식

prompt = PromptTemplate.from_template(template)

chain = prompt | gpu_llm | StrOutputParser()

question = "대한민국의 수도는 어디야?"

print(chain.invoke({"question":question}))