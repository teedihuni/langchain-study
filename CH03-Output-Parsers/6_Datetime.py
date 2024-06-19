from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser

load_dotenv()
logging.langsmith("CH03-Output-Parsers")

output_parser = DatetimeOutputParser()

template = """Answer the users question:

{question}

{format_instructions}"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions":output_parser.get_format_instructions()
    }
)

print(prompt)

chain = (
    prompt | OpenAI() | output_parser
)

output = chain.invoke({"question" : "Google 이 창업한 연도는?"})
print(output)