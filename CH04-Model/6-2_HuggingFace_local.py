from langchain_huggingface import HuggingFaceEndpoint
#from langchain_community.llms import huggingface_hub # 기존의 튜토리얼과 다른 부분임
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
logging.langsmith('CH04-Model')
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"
os.environ["HF_HOME"] = "/home/dhlee2/workspace/my_github/LangChain/CH04-Model"

# HuggingFace Repository ID
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"


# HuggingFaceHub 객체 생성
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature= 0.1,   # 샘플링 온도를 설정합니다. 값이 높을수록 더 다양한 출력을 생성합니다.
    max_new_tokens=2048,  # 생성할 최대 토큰 길이를 설정합니다.

    callbacks=[StreamingStdOutCallbackHandler()],  # 콜백을 설정합니다.
    streaming=True,
    task="text-generation",  # 텍스트 생성
)

template = """Summarizes TEXT in simple bullet points ordered from most important to least important.
TEXT:
{text}

KeyPoints: """

prompt = PromptTemplate.from_template(template)

llm_chain = prompt | llm

text = """A Large Language Model (LLM) like me, ChatGPT, is a type of artificial intelligence (AI) model designed to understand, generate, and interact with human language. These models are "large" because they're built from vast amounts of text data and have billions or even trillions of parameters. Parameters are the aspects of the model that are learned from training data; they are essentially the internal settings that determine how the model interprets and generates language. LLMs work by predicting the next word in a sequence given the words that precede it, which allows them to generate coherent and contextually relevant text based on a given prompt. This capability can be applied in a variety of ways, from answering questions and composing emails to writing essays and even creating computer code. The training process for these models involves exposing them to a diverse array of text sources, such as books, articles, and websites, allowing them to learn language patterns, grammar, facts about the world, and even styles of writing. However, it's important to note that while LLMs can provide information that seems knowledgeable, their responses are generated based on patterns in the data they were trained on and not from a sentient understanding or awareness. The development and deployment of LLMs raise important considerations regarding accuracy, bias, ethical use, and the potential impact on various aspects of society, including employment, privacy, and misinformation. Researchers and developers continue to work on ways to address these challenges while improving the models' capabilities and applications."""

response = llm_chain.invoke(input = text)
print(response)