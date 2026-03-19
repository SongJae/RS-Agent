"""
Transformers Backend

로컬 경로의 HuggingFace 모델을 사용하는 실제 추론 백엔드.
Function calling을 지원하는 instruction-tuned 모델이 필요합니다.

권장 모델 (tool calling 지원):
  - Qwen2.5-7B-Instruct   (14GB RAM / FP16)
  - Llama-3.1-8B-Instruct (16GB RAM / FP16)
  - Phi-3-mini-4k-instruct (8GB RAM / FP16)

폐쇄망 모델 준비 방법:
  인터넷 환경에서 실행:
    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output /models/Qwen2.5-7B-Instruct

  폐쇄망에서 실행:
    python scripts/start_local_server.py --backend transformers --model-path /models/Qwen2.5-7B-Instruct
"""

import json
import uuid
import time
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _build_prompt(messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    """
    메시지 리스트를 단일 프롬프트 문자열로 변환합니다.
    모델이 apply_chat_template을 지원하지 않을 때 사용하는 폴백입니다.
    """
    parts = []
    if tools:
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        parts.append(
            f"You have access to the following tools:\n{tools_json}\n\n"
            "To call a tool, respond with JSON in this format:\n"
            '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n'
        )
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"Human: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
        elif role == "tool":
            parts.append(f"Tool Result: {content}\n")
    parts.append("Assistant:")
    return "".join(parts)


def _parse_tool_call(text: str, tools: List[Dict]) -> Optional[Dict]:
    """생성된 텍스트에서 tool call JSON을 파싱합니다."""
    # {"tool_call": ...} 패턴 추출
    match = re.search(r'\{.*?"tool_call".*?\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        tc = data.get("tool_call", {})
        name = tc.get("name", "")
        tool_names = {t["function"]["name"] for t in tools}
        if name in tool_names:
            return {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return None


class TransformersBackend:
    """
    HuggingFace Transformers 기반 로컬 LLM 백엔드.
    완전 오프라인 환경에서 동작합니다.
    """

    def __init__(self, model_path: str, device: str = "auto", load_in_8bit: bool = False):
        """
        Args:
            model_path: 로컬 모델 디렉토리 경로 (예: /models/Qwen2.5-7B-Instruct)
            device: 사용 디바이스 ("auto", "cpu", "cuda")
            load_in_8bit: 8비트 양자화 사용 여부 (GPU 메모리 절약)
        """
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.model_path = model_path
        self._load_model(device, load_in_8bit)

    def _load_model(self, device: str, load_in_8bit: bool):
        """모델과 토크나이저를 로컬 경로에서 로드합니다."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading model from {self.model_path} (offline mode)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        load_kwargs: Dict = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        if device not in ("auto",) and not load_in_8bit:
            self.model = self.model.to(device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def chat_completion(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> Dict:
        """OpenAI 호환 chat completion 응답을 생성합니다."""
        import torch

        msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        # apply_chat_template 시도 (Qwen2, Llama3 등 지원 모델)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = _build_prompt(messages, tools)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # tool call 파싱 시도
        tool_call = None
        if tools:
            tool_call = _parse_tool_call(generated, tools)

        if tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": generated, "tool_calls": None}
            finish_reason = "stop"

        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = len(new_tokens)

        return {
            "id": msg_id,
            "object": "chat.completion",
            "created": ts,
            "model": self.model_path,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
