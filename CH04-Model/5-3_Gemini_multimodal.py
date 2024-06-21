from dotenv import load_dotenv
load_dotenv()

import os
print(f"[API KEY]\n{os.environ['GOOGLE_API_KEY']}")

from langchain_teddynote import logging
logging.langsmith('CH04-Model')

from langchain_google_genai import ChatGoogleGenerativeAI,HarmBlockThreshold,HarmCategory
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import base64
from PIL import Image
from io import BytesIO

# 이미지 URL을 지정합니다.
image_url = "https://picsum.photos/seed/picsum/300/300"

# 지정된 URL에서 이미지 데이터를 가져옵니다.
response = requests.get(image_url)
img_data = response.content

# 이미지를 불러옵니다.
img = Image.open(BytesIO(img_data))

# 이미지를 표시합니다.
img.show()


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
    # HTML을 렌더링하여 이미지 표시


file_path = "./images/jeju-beach.jpg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)
plt_img_base64(image_b64)
# Google의 Gemini-pro-vision 모델을 사용하여 ChatGoogleGenerativeAI 객체를 생성합니다.
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# 예시
message = HumanMessage(  # 사용자 메시지를 생성합니다.
    content=[
        {
            "type": "text",  # 메시지 유형을 텍스트로 지정합니다.
            "text": "What's in this image?",  # 이미지에 대한 질문을 텍스트로 입력합니다.
        },  # 선택적으로 텍스트 부분을 제공할 수 있습니다.
        # 메시지 유형을 이미지 URL로 지정하고, 이미지 URL을 입력합니다.
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
    ]
)
print(llm.invoke([message]))  # 생성된 사용자 메시지를 모델에 전달하여 응답을 생성합니다.
