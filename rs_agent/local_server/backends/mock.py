"""
Mock LLM Backend

실제 모델 없이 RS-Agent의 tool calling 파이프라인 전체를 검증합니다.
키워드 기반으로 적절한 tool을 선택하고, 올바른 OpenAI 메시지 포맷을 반환합니다.

실제 배포에서는 TransformersBackend 또는 외부 Ollama/vLLM 서버를 사용하세요.
"""

import json
import re
import uuid
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── 쿼리 키워드 → 도구 매핑 (구체적인 규칙을 먼저 배치) ──────
_TOOL_RULES: List[Dict] = [
    # ── 지식 질의 (이미지 불필요, 가장 먼저 체크) ────────────
    {
        "keywords": [
            "knowledge", "domain", "explain", "what is", "what are",
            "원격탐사란", "설명", "알려줘", "무엇", "어떻게", "지식",
        ],
        "tool": "knowledge_query",
        "args_template": {"query": "{query}"},
        "summary": "도메인 지식 검색이 완료되었습니다. 관련 전문 지식이 반환되었습니다.",
    },
    # ── SAR 관련 (sar 키워드 우선) ────────────────────────────
    {
        "keywords": ["sar", "synthetic aperture", "합성개구레이더", "레이더 영상", "선박 탐지"],
        "tool": "sar_object_detection",
        "args_template": {"image_path": "{image_path}", "categories": ["ship", "vehicle"]},
        "summary": "SAR 영상 객체 탐지가 완료되었습니다. 마이크로파 산란 특성을 이용한 탐지 결과입니다.",
    },
    # ── 이미지 향상 ──────────────────────────────────────────
    {
        "keywords": ["cloud", "구름"],
        "tool": "cloud_removal",
        "args_template": {"image_path": "{image_path}", "output_path": "output_cloud_removed.jpg"},
        "summary": "구름 제거 처리가 완료되었습니다. 가시성이 크게 향상되었으며 지표면 분석이 가능한 상태입니다.",
    },
    {
        "keywords": ["haze", "dehaze", "안개", "연무"],
        "tool": "image_dehazing",
        "args_template": {"image_path": "{image_path}", "output_path": "output_dehazed.jpg"},
        "summary": "헤이즈 제거가 완료되었습니다. 대기 산란 효과가 제거되어 영상 선명도가 향상되었습니다.",
    },
    {
        "keywords": ["super resolution", "super_resolution", "upscale", "초해상도", "해상도 향상", "4배"],
        "tool": "super_resolution",
        "args_template": {"image_path": "{image_path}", "scale_factor": 4, "output_path": "output_sr.jpg"},
        "summary": "초해상도 처리가 완료되었습니다. 공간 해상도가 4배 향상되었습니다.",
    },
    {
        "keywords": ["denoise", "noise", "노이즈"],
        "tool": "image_denoising",
        "args_template": {"image_path": "{image_path}", "output_path": "output_denoised.jpg"},
        "summary": "노이즈 제거가 완료되었습니다. 신호 대 잡음비(SNR)가 향상되었습니다.",
    },
    # ── 특징 추출 (일반 탐지보다 우선) ───────────────────────
    {
        "keywords": ["building", "건물", "건축물", "빌딩"],
        "tool": "building_extraction",
        "args_template": {"image_path": "{image_path}", "output_path": "output_buildings.geojson"},
        "summary": "건물 추출이 완료되었습니다. 건물 풋프린트가 GeoJSON 형태로 저장되었습니다.",
    },
    {
        "keywords": ["road", "도로", "도로망", "네트워크"],
        "tool": "road_extraction",
        "args_template": {"image_path": "{image_path}", "output_path": "output_roads.geojson"},
        "summary": "도로망 추출이 완료되었습니다. 도로 중심선 데이터가 GeoJSON 형태로 저장되었습니다.",
    },
    # ── 항공기 ────────────────────────────────────────────────
    {
        "keywords": ["aircraft", "airplane", "plane", "항공기", "비행기", "기종"],
        "tool": "aircraft_classification_optical",
        "args_template": {"image_path": "{image_path}"},
        "summary": "항공기 분류가 완료되었습니다. 기종 및 유형이 식별되었습니다.",
    },
    # ── 분류·분석 ─────────────────────────────────────────────
    {
        "keywords": ["land use", "land cover", "lulc", "토지이용", "토지 피복"],
        "tool": "land_use_classification",
        "args_template": {"image_path": "{image_path}", "output_path": "output_lulc.png"},
        "summary": "토지이용 분류가 완료되었습니다. LULC 지도가 생성되었습니다.",
    },
    {
        "keywords": ["change", "변화", "차이", "before", "after"],
        "tool": "change_detection",
        "args_template": {"image_path_before": "{image_path}", "image_path_after": "after.jpg"},
        "summary": "변화 탐지가 완료되었습니다. 두 시기 영상 간의 토지 피복 변화가 감지되었습니다.",
    },
    {
        "keywords": ["damage", "disaster", "피해", "재해", "재난"],
        "tool": "damage_assessment",
        "args_template": {"image_path": "{image_path}", "disaster_type": "flood"},
        "summary": "피해 평가가 완료되었습니다. 영향 지역 및 피해 수준이 분석되었습니다.",
    },
    {
        "keywords": ["count", "counting", "개수", "몇 개", "몇개"],
        "tool": "object_counting",
        "args_template": {"image_path": "{image_path}", "object_type": "vehicle"},
        "summary": "객체 계수가 완료되었습니다. 영상 내 대상 객체의 수량이 산정되었습니다.",
    },
    {
        "keywords": ["segment", "segmentation", "픽셀", "의미론적"],
        "tool": "semantic_segmentation",
        "args_template": {"image_path": "{image_path}", "output_path": "output_segmentation.png"},
        "summary": "시맨틱 분할이 완료되었습니다. 픽셀 단위의 토지 피복 분류 결과가 생성되었습니다.",
    },
    {
        "keywords": ["scene", "classify", "classification", "장면", "분류"],
        "tool": "scene_classification",
        "args_template": {"image_path": "{image_path}"},
        "summary": "장면 분류가 완료되었습니다. 원격탐사 영상의 토지 이용 유형이 분류되었습니다.",
    },
    {
        "keywords": ["rotated", "oriented", "obb"],
        "tool": "rotated_object_detection",
        "args_template": {"image_path": "{image_path}", "categories": ["vehicle", "ship"]},
        "summary": "회전 객체 탐지가 완료되었습니다. 방향 정보를 포함한 OBB(Oriented Bounding Box) 결과가 반환되었습니다.",
    },
    # ── 일반 탐지 (마지막에 배치) ─────────────────────────────
    {
        "keywords": ["detect", "detection", "object", "탐지", "검출", "선박", "차량", "ship", "vehicle"],
        "tool": "horizontal_object_detection",
        "args_template": {"image_path": "{image_path}", "categories": ["vehicle", "ship", "aircraft"]},
        "summary": "객체 탐지가 완료되었습니다. 여러 개의 객체가 검출되었으며 위치 및 클래스 정보가 반환되었습니다.",
    },
    {
        "keywords": ["vqa", "visual question", "what", "how many", "영상 질문"],
        "tool": "remote_sensing_vqa",
        "args_template": {"image_path": "{image_path}", "question": "{question}"},
        "summary": "원격탐사 VQA 분석이 완료되었습니다. 영상 내용에 기반한 답변이 생성되었습니다.",
    },
    {
        "keywords": ["road", "도로", "네트워크"],
        "tool": "road_extraction",
        "args_template": {"image_path": "{image_path}", "output_path": "output_roads.geojson"},
        "summary": "도로망 추출이 완료되었습니다. 도로 중심선 데이터가 GeoJSON 형태로 저장되었습니다.",
    },
    {
        "keywords": ["aircraft", "airplane", "plane", "항공기", "비행기"],
        "tool": "aircraft_classification_optical",
        "args_template": {"image_path": "{image_path}"},
        "summary": "항공기 분류가 완료되었습니다. 기종 및 유형이 식별되었습니다.",
    },
    {
        "keywords": ["damage", "disaster", "피해", "재해"],
        "tool": "damage_assessment",
        "args_template": {"image_path": "{image_path}", "disaster_type": "flood"},
        "summary": "피해 평가가 완료되었습니다. 영향 지역 및 피해 수준이 분석되었습니다.",
    },
    {
        "keywords": ["count", "counting", "개수", "몇 개", "몇개"],
        "tool": "object_counting",
        "args_template": {"image_path": "{image_path}", "object_type": "vehicle"},
        "summary": "객체 계수가 완료되었습니다. 영상 내 대상 객체의 수량이 산정되었습니다.",
    },
    {
        "keywords": ["change", "변화", "차이"],
        "tool": "change_detection",
        "args_template": {"image_path_before": "{image_path}", "image_path_after": "after.jpg"},
        "summary": "변화 탐지가 완료되었습니다. 두 시기 영상 간의 토지 피복 변화가 감지되었습니다.",
    },
    {
        "keywords": ["land use", "land cover", "lulc", "토지"],
        "tool": "land_use_classification",
        "args_template": {"image_path": "{image_path}", "output_path": "output_lulc.png"},
        "summary": "토지이용 분류가 완료되었습니다. LULC 지도가 생성되었습니다.",
    },
    {
        "keywords": ["vqa", "question", "what", "how", "why", "where", "질문"],
        "tool": "remote_sensing_vqa",
        "args_template": {"image_path": "{image_path}", "question": "{question}"},
        "summary": "원격탐사 VQA 분석이 완료되었습니다. 영상 내용에 기반한 답변이 생성되었습니다.",
    },
    {
        "keywords": ["knowledge", "domain", "explain", "what is", "원격탐사란", "지식"],
        "tool": "knowledge_query",
        "args_template": {"query": "{query}"},
        "summary": "도메인 지식 검색이 완료되었습니다. 관련 전문 지식이 반환되었습니다.",
    },
]


def _extract_image_path(messages: List[Dict]) -> str:
    """메시지에서 이미지 경로 추출."""
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            m = re.search(r'[\w./\-]+\.(jpg|jpeg|png|tif|tiff)', content, re.IGNORECASE)
            if m:
                return m.group(0)
    return "input_image.jpg"


def _select_tool(messages: List[Dict], tools: List[Dict]) -> Optional[Dict]:
    """
    user 메시지 키워드를 기반으로 사용할 tool을 선택.
    system 메시지는 제외 (시스템 프롬프트에 포함된 도메인 키워드와 혼동 방지).
    """
    if not tools:
        return None

    # user / tool 결과 메시지만 참조 (system 프롬프트 제외)
    user_text = " ".join(
        msg.get("content", "") for msg in messages
        if msg.get("role") in ("user",) and isinstance(msg.get("content"), str)
    ).lower()

    tool_names = {t["function"]["name"] for t in tools}

    for rule in _TOOL_RULES:
        if any(kw in user_text for kw in rule["keywords"]):
            if rule["tool"] in tool_names:
                return rule
    # 기본값: horizontal_object_detection
    return next(r for r in _TOOL_RULES if r["tool"] == "horizontal_object_detection")


def _fill_args(template: Dict, image_path: str, messages: List[Dict]) -> Dict:
    """템플릿 인자를 실제 값으로 채움."""
    full_text = " ".join(
        msg.get("content", "") for msg in messages
        if isinstance(msg.get("content"), str)
    )
    filled = {}
    for k, v in template.items():
        if isinstance(v, str):
            v = v.replace("{image_path}", image_path)
            v = v.replace("{query}", full_text[:200])
            v = v.replace("{question}", full_text[:200])
        filled[k] = v
    return filled


class MockBackend:
    """
    규칙 기반 Mock LLM Backend.
    실제 모델 없이 RS-Agent tool calling 파이프라인을 검증합니다.
    """

    def __init__(self):
        self._tool_call_state: Dict[str, Dict] = {}  # call_id → pending tool info
        logger.info("MockBackend initialized (no model required)")

    def chat_completion(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        """
        OpenAI chat.completions.create 응답 형태를 반환합니다.

        첫 번째 호출: tool_call 반환
        tool_result 포함 호출: 최종 답변 반환
        """
        msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        # tool_result가 있으면 최종 답변 단계
        has_tool_result = any(m.get("role") == "tool" for m in messages)

        if has_tool_result or not tools:
            return self._final_answer(messages, msg_id, ts)
        else:
            return self._tool_call_response(messages, tools, msg_id, ts)

    def _tool_call_response(
        self, messages: List[Dict], tools: List[Dict], msg_id: str, ts: int
    ) -> Dict:
        """Tool call 응답 생성."""
        rule = _select_tool(messages, tools)
        image_path = _extract_image_path(messages)
        tool_args = _fill_args(rule["args_template"], image_path, messages)
        call_id = f"call_{uuid.uuid4().hex[:12]}"

        self._tool_call_state[call_id] = rule

        logger.info(f"MockBackend: selecting tool '{rule['tool']}'")
        return {
            "id": msg_id,
            "object": "chat.completion",
            "created": ts,
            "model": "mock-rs-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": rule["tool"],
                                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }

    def _final_answer(self, messages: List[Dict], msg_id: str, ts: int) -> Dict:
        """Tool 결과를 받아 최종 자연어 답변 생성."""
        # 가장 최근 tool 결과 파싱
        tool_result_content = ""
        tool_name = ""
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                tool_result_content = msg.get("content", "")
                break
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                tc = msg.get("tool_calls", [])
                if tc:
                    tool_name = tc[0].get("function", {}).get("name", "")
                break

        try:
            result_data = json.loads(tool_result_content)
            status = result_data.get("status", "unknown")
        except Exception:
            status = "success"

        # 해당 도구의 요약 문구 조회
        summary = next(
            (r["summary"] for r in _TOOL_RULES if r["tool"] == tool_name),
            "처리가 완료되었습니다.",
        )

        user_query = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"
             and isinstance(m.get("content"), str)),
            "요청",
        )

        answer = (
            f"분석 완료: **{tool_name}** 도구를 사용하여 요청을 처리했습니다.\n\n"
            f"{summary}\n\n"
            f"**처리 상태**: {status}\n"
            f"**원본 요청**: {user_query[:100]}"
        )

        return {
            "id": msg_id,
            "object": "chat.completion",
            "created": ts,
            "model": "mock-rs-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer, "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 60, "total_tokens": 140},
        }
