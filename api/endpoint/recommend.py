from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from ai.ai_recommend import recommend_ai

router = APIRouter()


class RecommendRequest(BaseModel):
    user_input: str


class RecommendResponse(BaseModel):
    recommendation: str
    links: Dict[str, str]


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_endpoint(request: RecommendRequest):
    """
    사용자 입력을 받아 적합한 AI를 추천하고 관련 링크를 반환합니다.
    """
    try:
        result = recommend_ai(request.user_input)
        return RecommendResponse(
            recommendation=result["recommendation"],
            links=result["links"]
        )
    
    except ValueError as e:
        # 금지어 체크 실패
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # 기타 오류
        raise HTTPException(status_code=500, detail=f"AI 추천 중 오류가 발생했습니다: {str(e)}")
