from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 금지어 리스트
GUMJI_WORDS = ['노무현', '이기', "운지", "이기", "대중", "노무", "DJ", "찢",
               "앰뒤", "응디", "느그", "느금", "애미", "시발", "씨발", "개새끼",
               "개쌔키", "개새키", "미친", "새끼", "또라이", "간나", "지랄", "부엉이 바위"]

# 시스템 프롬프트
SYSTEM_PROMPT = """
-두 줄 이상으로 짜줘 예를 들지말고 그 상황에 맞는 프롬프트를 출력해줘
-너는 클라이언트에게 프롬프트에 대한 예시를 출력하지말고 너가 클라이언트가 입력한 상황에 맞춰서 프롬프트를 짜줘
-클라이언트는 지금부터 상황을 입력 할 예정이고 너는 지금부터 그 상황에 인공지능이 잘 답변 할 수 있도록 프롬프트를 만들어야해
-클라이언트에게 더 긴 프롬프트를 출력해
-예를 들지말고 너가 프롬프트를 직접 작성해줘
- 욕설이나 성적이거나 정치적인 내용이 들어오면 '이 내용에는 답변 할 수 없습니다' 라고 해줘
"""


def check_forbidden_words(text: str) -> bool:
    """금지어가 포함되어 있는지 확인"""
    return any(word in text for word in GUMJI_WORDS)


def generate_prompt(user_input: str) -> str:
    """사용자 입력을 받아 AI 프롬프트를 생성"""
    if check_forbidden_words(user_input):
        raise ValueError("이 내용에는 답변 할 수 없습니다.")
    
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::CWLHAQEj",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    
    return response.choices[0].message.content


def main():
    """CLI 인터페이스"""
    print("AI에게 상황을 입력해주세요! (종료하려면 'quit' 입력)")
    
    while True:
        user_input = input("\n클라이언트: ")
        
        if user_input.lower() in ["q", "종료"]:
            print("대화를 종료합니다.")
            break
        
        try:
            result = generate_prompt(user_input)
            print("AI:", result)
        except ValueError as e:
            print("AI:", str(e))
        except Exception as e:
            print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
