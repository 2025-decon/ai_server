from openai import OpenAI
import re
from typing import Dict, List
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# AI 링크 매핑
AI_LINKS = {
    "gpt": "https://chatgpt.com/",
    "claude": "https://claude.ai/new",
    "gemini": "https://www.google.com/ai/gemini",
    "huggingface": "https://huggingface.co/",
    "llama": "https://github.com/ollama/ollama"
}

# 금지어 리스트
GUMJI_WORDS = [
    "노무현", "이기", "운지", "대중", "DJ", "앰뒤",
    "응디", "느그", "느금", "애미", "시발", "씨발",
    "개새끼", "개쌔키", "미친", "새끼", "또라이", "간나", "지랄",
    "이재명", "문재인", "문제인", "박근혜", "박근헤", "박근해", "이승만", "전두환"
]

# 시스템 프롬프트
SYSTEM_PROMPT = (
    "너는 AI추천 전문가야. "
    "너는 'gemini', 'Claude', 'gpt' 3개중에서 골라서 클라이언트에게 출력해야해 "
    "만약에 커피 어떻게 만드냐는 입력이 들어오면 이상한 'coffee agent'나 'Culinary Agent ' 같은거 절대 출력하지마"
    "절대로 학습 데이터를 그대로 복사하거나 그대로 출력하지 마. "
    "클라이언트가 영어로 질문해도 새로운 문장으로 응답해. "
    "코드 작성 요청이면 코드 작성에 적합한 인공지능을 추천해. "
    "그 외에는 상황에 맞는 인공지능을 제안해."
)


def sanitize_output(text: str) -> str:
    """학습 내용 그대로 복사 방지"""
    text = re.sub(r'[{]"prompt".*?}', '', text, flags=re.DOTALL)
    text = re.sub(r'[{]"completion".*?}', '', text, flags=re.DOTALL)
    text = re.sub(r'(\b.+\b)( \1)+', r'\1', text)
    text = re.sub(r'["]{3,}.*?["]{3,}', '', text, flags=re.DOTALL)
    text = text.strip()
    return text


def check_forbidden_words(text: str) -> bool:
    """금지어가 포함되어 있는지 확인"""
    return any(word in text for word in GUMJI_WORDS)


def extract_links(recommendation_text: str) -> Dict[str, str]:
    """추천 텍스트에서 AI 링크 추출"""
    links = {}
    for ai_name, url in AI_LINKS.items():
        if ai_name.lower() in recommendation_text.lower():
            links[ai_name] = url
    return links


def recommend_ai(user_input: str) -> Dict[str, any]:
    """사용자 입력을 받아 AI 추천 결과를 반환"""
    if check_forbidden_words(user_input):
        raise ValueError("이 내용에는 답변할 수 없습니다.")
    
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::CX5bDXxB",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    
    recommendation_text = response.choices[0].message.content
    recommendation_text = sanitize_output(recommendation_text)
    
    # 학습 데이터 유사성 체크
    if any(key in recommendation_text for key in ['"prompt":', '"completion":', '{', '}']):
        recommendation_text = "학습 데이터와 유사한 내용이 감지되어 새로운 추천 문장으로 대체합니다. 이 상황에는 GPT를 추천합니다."
    
    # 관련 링크 추출
    links = extract_links(recommendation_text)
    
    return {
        "recommendation": recommendation_text,
        "links": links
    }


def main():
    """CLI 인터페이스"""
    print("AI 추천! (종료하려면 'quit' 입력)")
    
    while True:
        user_input = input("\n클라이언트: ")
        
        if user_input.lower() in ["q", "quit", "종료"]:
            print("대화를 종료합니다.")
            break
        
        try:
            result = recommend_ai(user_input)
            print("AI:", result["recommendation"])
            for ai_name, url in result["links"].items():
                print(f"추천 링크 ({ai_name}): {url}")
        except ValueError as e:
            print("AI:", str(e))
        except Exception as e:
            print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
