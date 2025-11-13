from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai.prompt_aitest import generate_prompt as ai_generate_prompt

router = APIRouter()


class PromptRequest(BaseModel):
    user_input: str


class PromptResponse(BaseModel):
    prompt: str


@router.post("/prompt", response_model=PromptResponse)
async def generate_prompt_endpoint(request: PromptRequest):
    """
    사용자 입력을 받아 AI가 생성한 프롬프트를 반환합니다.
    """
    try:
        generated_prompt = ai_generate_prompt(request.user_input)
        return PromptResponse(prompt=generated_prompt)
    
    except ValueError as e:
        # 금지어 체크 실패
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # 기타 오류
        raise HTTPException(status_code=500, detail=f"AI 프롬프트 생성 중 오류가 발생했습니다: {str(e)}")
