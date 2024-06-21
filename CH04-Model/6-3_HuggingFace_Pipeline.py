from langchain_huggingface import HuggingFaceEndpoint
#from langchain_community.llms import huggingface_hub # 기존의 튜토리얼과 다른 부분임
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
logging.langsmith('CH04-Model')
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"
os.environ["HF_HOME"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"


## 모델 부르는 방법 1 : 
# hf = HuggingFacePipeline.from_model_id(
#     model_id = "beomi/llama-2-ko-7b",
#     task = 'text-generation',
#     pipeline_kwargs={'max_new_kokens': 512},
# )

## 모델 부르는 방법 2 : transformer 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = 'beomi/llama-2-ko-7b'
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline('text-generation', model=model,
                tokenizer=tokenizer, max_new_tokens =512)

hf = HuggingFacePipeline(pipeline=pipe)

## Create Chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Answer the following question in Korean.
#Question:
{question}

#Answer: """ # 질문 답변 형식

prompt = PromptTemplate.from_template(template)

chain = prompt | hf | StrOutputParser()

question = "대한민국의 수도는 어디야?"

print(chain.invoke({"question":question}))