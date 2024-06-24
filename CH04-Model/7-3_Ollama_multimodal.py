## vlm 모델인 LLaVA 모델 다운로드

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import base64
from io import BytesIO
from PIL import Image
import webbrowser
import os

from langchain_teddynote import logging

#logging.langsmith('CH04-model')


def convert_to_base64(pil_image):
    """
    PIL 이미지를 Base64로 인코딩된 문자열로 변환합니다.

    :param pil_image: PIL 이미지
    :return: 크기 조정된 Base64 문자열
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # 필요한 경우 형식을 변경할 수 있습니다.
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Base64로 인코딩된 문자열을 이미지로 표시합니다.

    :param img_base64:  Base64 문자열
    """
    # Base64 문자열을 소스로 사용하여 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
     # HTML 파일 생성
    html_content = f"""
    <html>
        <body>
            {image_html}
        </body>
    </html>
    """
    
    # HTML 파일 저장
    with open('image.html', 'w') as file:
        file.write(html_content)
    
    # 웹 브라우저로 HTML 파일 열기
    webbrowser.open('file://' + os.path.realpath('image.html'))

def prompt_func(data):  # 프롬프트 함수를 정의합니다.
    text = data["text"]  # 데이터에서 텍스트를 가져옵니다.
    image = data["image"]  # 데이터에서 이미지를 가져옵니다.

    image_part = {  # 이미지 부분을 정의합니다.
        "type": "image_url",  # 이미지 URL 타입을 지정합니다.
        "image_url": f"data:image/jpeg;base64,{image}",  # 이미지 URL을 생성합니다.
    }

    content_parts = []  # 콘텐츠 부분을 저장할 리스트를 초기화합니다.

    text_part = {"type": "text", "text": text}  # 텍스트 부분을 정의합니다.

    content_parts.append(image_part)  # 이미지 부분을 콘텐츠 부분에 추가합니다.
    content_parts.append(text_part)  # 텍스트 부분을 콘텐츠 부분에 추가합니다.

    return [HumanMessage(content=content_parts)]  # HumanMessage 객체를 반환합니다.


file_path = "./images/jeju-beach.jpg"
pil_image = Image.open(file_path)

## 이미지 변환 및 로딩
image_b64 = convert_to_base64(pil_image)
plt_img_base64(image_b64)

# 모델 정의
llm = ChatOllama(model='bakllava', temperature=0)

chain = prompt_func | llm | StrOutputParser()

result = chain.invoke(
    {'text':"Describe a picture in bullet points.", "image":image_b64}
)

print(result)
