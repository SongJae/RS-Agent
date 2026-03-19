"""
RS-Agent Local LLM Server (OpenAI 호환)

FastAPI 기반의 OpenAI-compatible 서버.
Ollama/vLLM 없이 폐쇄망에서 RS-Agent를 완전 운용할 수 있습니다.

엔드포인트:
  GET  /v1/models               - 사용 가능한 모델 목록
  POST /v1/chat/completions     - 채팅 완료 (tool calling 포함)
  GET  /health                  - 헬스 체크
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Pydantic 모델 (OpenAI API 스키마) ────────────────────────

class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[Message]
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = "auto"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


# ── App Factory ───────────────────────────────────────────────

def create_app(backend_name: str = "mock", **backend_kwargs) -> FastAPI:
    """
    FastAPI 앱을 생성합니다.

    Args:
        backend_name: "mock" 또는 "transformers"
        backend_kwargs: 백엔드 초기화 인자
            - mock: 없음
            - transformers: model_path, device, load_in_8bit
    """
    app = FastAPI(
        title="RS-Agent Local LLM Server",
        description="폐쇄망 운용을 위한 OpenAI 호환 로컬 LLM 서버",
        version="1.0.0",
    )

    # 백엔드 초기화
    if backend_name == "mock":
        from rs_agent.local_server.backends.mock import MockBackend
        backend = MockBackend()
        model_id = "mock-rs-agent"
    elif backend_name == "transformers":
        from rs_agent.local_server.backends.transformers_backend import TransformersBackend
        model_path = backend_kwargs.get("model_path", "")
        if not model_path:
            raise ValueError("transformers 백엔드는 model_path 인자가 필요합니다.")
        backend = TransformersBackend(
            model_path=model_path,
            device=backend_kwargs.get("device", "auto"),
            load_in_8bit=backend_kwargs.get("load_in_8bit", False),
        )
        model_id = model_path
    else:
        raise ValueError(f"알 수 없는 백엔드: {backend_name}. 'mock' 또는 'transformers'를 사용하세요.")

    app.state.backend = backend
    app.state.model_id = model_id
    logger.info(f"Local LLM server started: backend={backend_name}, model={model_id}")

    # ── 엔드포인트 ────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": backend_name, "model": model_id}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        tools = None
        if request.tools:
            tools = [
                {
                    "type": t.type,
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {},
                    },
                }
                for t in request.tools
            ]

        try:
            response = app.state.backend.chat_completion(
                messages=messages,
                tools=tools,
                max_new_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.1,
            )
            return JSONResponse(content=response)
        except Exception as e:
            logger.error(f"chat_completion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app
