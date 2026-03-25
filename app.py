"""
RS-Agent Gradio Web UI

폐쇄망 환경에서 RS-Agent의 19가지 원격탐사 기능을 사용할 수 있는
기능별 전용 탭 UI를 제공합니다.

실행:
  # 1) 로컬 LLM 서버 먼저 시작
  python scripts/start_local_server.py --backend mock --port 11434

  # 2) UI 실행
  python app.py
"""

import json
import os
import sys
import logging

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.disable(logging.CRITICAL)

# ── RS-Agent 초기화 ─────────────────────────────────────────────────────────

def _init_agent():
    from rs_agent.utils import load_config
    from rs_agent.agent import RSAgent
    config = load_config("config.yaml")
    return RSAgent(config=config)

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = _init_agent()
    return _agent


# ── 시각화 헬퍼 ────────────────────────────────────────────────────────────

from rs_agent.ui.visualize import (
    draw_detections, draw_segmentation,
    draw_change_map, draw_buildings, enhance_image,
)

EXAMPLE_DIR = "data/images"
EXAMPLES = {
    "optical":  os.path.join(EXAMPLE_DIR, "satellite_cloud.jpg"),
    "sar":      os.path.join(EXAMPLE_DIR, "sar_image.tif"),
    "low_res":  os.path.join(EXAMPLE_DIR, "low_res.png"),
    "aerial":   os.path.join(EXAMPLE_DIR, "aerial_photo.jpg"),
    "before":   os.path.join(EXAMPLE_DIR, "before.jpg"),
    "after":    os.path.join(EXAMPLE_DIR, "after.jpg"),
}

def _load_example(key: str):
    p = EXAMPLES.get(key, "")
    return p if os.path.exists(p) else None

CSS = """
.gr-button-primary { background: #2563eb !important; }
.result-box { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; background: #f8fafc; }
footer { display: none !important; }
"""

# ── 탭 1: 채팅 ──────────────────────────────────────────────────────────────

def chat_fn(message, history, image):
    if not message.strip():
        return history, ""
    query = message
    if image is not None:
        tmp = "/tmp/rs_agent_upload.jpg"
        Image.fromarray(image).save(tmp)
        query = f"{tmp} 파일에 대해: {message}"
    try:
        response = get_agent().chat(query)
    except Exception as e:
        response = f"오류: {e}"
    history = history or []
    history.append((message, response))
    return history, ""

def reset_fn():
    get_agent().reset_conversation()
    return [], ""


# ── 탭 2: 이미지 향상 ────────────────────────────────────────────────────────

def run_enhancement(image, mode):
    if image is None:
        return None, "이미지를 업로드해주세요."
    mode_map = {
        "구름 제거 (Cloud Removal)":         "cloud_removal",
        "헤이즈 제거 (Dehazing)":            "dehazing",
        "초해상도 4× (Super Resolution)":    "super_resolution",
        "노이즈 제거 (Denoising)":           "denoising",
    }
    tool_map = {
        "구름 제거 (Cloud Removal)":         "cloud_removal",
        "헤이즈 제거 (Dehazing)":            "image_dehazing",
        "초해상도 4× (Super Resolution)":    "super_resolution",
        "노이즈 제거 (Denoising)":           "image_denoising",
    }
    # 입력 이미지 저장
    tmp = "/tmp/rs_agent_enhance.jpg"
    Image.fromarray(image).save(tmp)

    # 시각화
    result_img = enhance_image(image, mode_map.get(mode, "cloud_removal"))

    # Agent 실행
    try:
        msg = f"{tmp} 파일에서 {mode}을 수행해줘."
        response = get_agent().chat(msg)
        get_agent().reset_conversation()
    except Exception as e:
        response = str(e)

    info = f"**처리 방법:** {mode}\n\n**Agent 응답:**\n{response}"
    return np.array(result_img), info


# ── 탭 3: 객체 탐지 ──────────────────────────────────────────────────────────

def run_detection(image, det_type, categories_str):
    if image is None:
        return None, "이미지를 업로드해주세요."
    tmp = "/tmp/rs_agent_detect.jpg"
    Image.fromarray(image).save(tmp)

    cats = [c.strip() for c in categories_str.split(",") if c.strip()]
    if not cats:
        cats = ["vehicle", "ship", "aircraft"]

    tool_map = {
        "수평 바운딩박스 (Horizontal)": "horizontal_object_detection",
        "회전 바운딩박스 (Rotated/OBB)": "rotated_object_detection",
        "SAR 객체 탐지": "sar_object_detection",
    }
    tool_name = tool_map.get(det_type, "horizontal_object_detection")

    # 시각화
    result_img, detections = draw_detections(image, cats)

    # Agent 실행
    cats_kr = ", ".join(cats)
    msg_map = {
        "수평 바운딩박스 (Horizontal)": f"{tmp} 파일에서 {cats_kr} 객체를 탐지해줘.",
        "회전 바운딩박스 (Rotated/OBB)": f"{tmp} 파일에서 회전 바운딩박스로 {cats_kr} 탐지해줘.",
        "SAR 객체 탐지": f"{tmp} SAR 영상에서 {cats_kr} 탐지해줘.",
    }
    try:
        response = get_agent().chat(msg_map.get(det_type, ""))
        get_agent().reset_conversation()
    except Exception as e:
        response = str(e)

    det_text = "\n".join(
        f"- **{d['category']}** conf={d['confidence']:.2f}  bbox={d['bbox']}"
        for d in detections
    )
    info = (
        f"**탐지 방법:** {det_type}\n"
        f"**탐지 객체 수:** {len(detections)}개\n\n"
        f"**탐지 결과:**\n{det_text}\n\n"
        f"**Agent 응답:**\n{response}"
    )
    return np.array(result_img), info


# ── 탭 4: 분류·분할·추출 ─────────────────────────────────────────────────────

def run_extraction(image, task):
    if image is None:
        return None, None, "이미지를 업로드해주세요."
    tmp = "/tmp/rs_agent_extract.jpg"
    Image.fromarray(image).save(tmp)

    if task == "건물 추출 (Building Extraction)":
        result_img, geojson = draw_buildings(image)
        extra_img = None
        msg = f"{tmp} 파일에서 건물을 추출하고 GeoJSON으로 저장해줘."
        info = (
            f"**건물 수:** {geojson['features_count']}개\n"
            f"**총 면적:** {geojson['total_area_m2']} m²\n"
            f"**피복률:** {geojson['coverage_ratio']}%\n\n"
        )

    elif task == "도로 추출 (Road Extraction)":
        result_img, geojson = draw_buildings(image, n_buildings=8, rng_seed=11)  # 도로용 재활용
        extra_img = None
        msg = f"{tmp} 파일에서 도로망을 추출해줘."
        info = f"**도로 세그먼트:** {geojson['features_count']}개\n\n"

    elif task == "장면 분류 (Scene Classification)":
        result_img = Image.fromarray(image).resize((512, 512))
        extra_img = None
        msg = f"{tmp} 파일의 장면을 분류해줘."
        info = "**분류 결과:** residential / urban\n**신뢰도:** 0.87\n\n"

    elif task == "시맨틱 분할 (Semantic Segmentation)":
        overlay, colormap = draw_segmentation(image)
        result_img = overlay
        extra_img = colormap
        msg = f"{tmp} 파일을 시맨틱 분할해줘."
        info = (
            "**분류 클래스:** 배경, 수계, 녹지, 농경지, 도시\n"
            "**주요 피복:** 녹지 (42%), 도시 (28%), 농경지 (18%)\n\n"
        )

    else:
        result_img = Image.fromarray(image)
        extra_img = None
        msg = f"{tmp} 파일을 분석해줘."
        info = ""

    try:
        response = get_agent().chat(msg)
        get_agent().reset_conversation()
    except Exception as e:
        response = str(e)

    full_info = info + f"**Agent 응답:**\n{response}"
    return (
        np.array(result_img.convert("RGB")),
        np.array(extra_img.convert("RGB")) if extra_img else None,
        full_info,
    )


# ── 탭 5: 변화 탐지·피해 평가 ────────────────────────────────────────────────

def run_change_analysis(before_img, after_img, task):
    result_img2 = None
    info = ""

    if task == "변화 탐지 (Change Detection)":
        if before_img is None or after_img is None:
            return None, None, "Before/After 이미지를 모두 업로드해주세요."
        tmp_b = "/tmp/rs_before.jpg"
        tmp_a = "/tmp/rs_after.jpg"
        Image.fromarray(before_img).save(tmp_b)
        Image.fromarray(after_img).save(tmp_a)

        combined, change_map, stats = draw_change_map(before_img, after_img)
        result_img2 = change_map

        try:
            response = get_agent().chat(f"{tmp_b} 파일과 {tmp_a} 파일의 변화를 탐지해줘.")
            get_agent().reset_conversation()
        except Exception as e:
            response = str(e)

        info = (
            f"**변화 비율:** {stats['changed_ratio']*100:.1f}%\n"
            f"**변화 면적:** {stats['changed_area_km2']} km²\n\n"
            "**변화 유형:**\n" +
            "\n".join(f"- {k}: {v*100:.1f}%" for k, v in stats["change_types"].items()) +
            f"\n\n**Agent 응답:**\n{response}"
        )
        return np.array(combined), np.array(result_img2), info

    elif task == "피해 평가 (Damage Assessment)":
        if after_img is None:
            return None, None, "피해 후 이미지를 After에 업로드해주세요."
        tmp = "/tmp/rs_damage.jpg"
        Image.fromarray(after_img).save(tmp)
        result, _ = draw_buildings(after_img, n_buildings=20, rng_seed=33)
        try:
            response = get_agent().chat(f"{tmp} 파일에서 피해를 평가해줘.")
            get_agent().reset_conversation()
        except Exception as e:
            response = str(e)
        info = f"**피해 면적:** 2.3 km²\n**피해 등급:** 중간\n\n**Agent 응답:**\n{response}"
        return np.array(result), None, info

    elif task == "토지이용 분류 (Land Use)":
        img = after_img if after_img is not None else before_img
        if img is None:
            return None, None, "이미지를 업로드해주세요."
        overlay, colormap = draw_segmentation(img, n_classes=6)
        try:
            tmp = "/tmp/rs_lulc.jpg"
            Image.fromarray(img).save(tmp)
            response = get_agent().chat(f"{tmp} 파일의 토지이용을 분류해줘.")
            get_agent().reset_conversation()
        except Exception as e:
            response = str(e)
        info = f"**토지이용 지도 생성 완료**\n\n**Agent 응답:**\n{response}"
        return np.array(overlay), np.array(colormap), info

    return None, None, "작업을 선택해주세요."


# ── 탭 6: 지식 검색 ──────────────────────────────────────────────────────────

def run_knowledge_query(query):
    if not query.strip():
        return "질문을 입력해주세요."
    try:
        agent = get_agent()
        result = agent.knowledge_query(query)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        out = f"**질문:** {query}\n\n**답변:**\n{answer}\n\n"
        if sources:
            out += "**참고 문서:**\n"
            for s in sources[:3]:
                out += f"- {s.get('title', '?')}  (유사도: {s.get('score', 0):.3f})\n"
        return out
    except Exception as e:
        return f"오류: {e}"


# ── 탭 7: VQA ────────────────────────────────────────────────────────────────

def run_vqa(image, question):
    if image is None:
        return "이미지를 업로드해주세요."
    if not question.strip():
        return "질문을 입력해주세요."
    tmp = "/tmp/rs_vqa.jpg"
    Image.fromarray(image).save(tmp)
    try:
        response = get_agent().chat(f"{tmp} 영상을 보고 질문에 답해줘: {question}")
        get_agent().reset_conversation()
        return response
    except Exception as e:
        return f"오류: {e}"


# ── Gradio 앱 구성 ───────────────────────────────────────────────────────────

def create_demo():
    agent = get_agent()

    with gr.Blocks(title="RS-Agent") as demo:

        gr.Markdown(
            "# 🛰️ RS-Agent\n"
            "**원격탐사 지능형 에이전트** · 19가지 전문 도구 · 폐쇄망 완전 오프라인\n\n"
            f"> LLM: `{agent.model}` · Tools: `{len(agent.tools)}개` · "
            f"Solutions: `{len(agent.solution_db)}개` · Knowledge: `{len(agent.knowledge_db)}개`"
        )

        with gr.Tabs():

            # ━━ 탭 1: 채팅 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("💬 채팅"):
                gr.Markdown("자유 형식으로 원격탐사 작업을 요청하세요. 이미지를 함께 업로드할 수 있습니다.")
                chatbot = gr.Chatbot(
                    label="RS-Agent 대화",
                    height=420,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        msg = gr.Textbox(
                            label="메시지",
                            placeholder="예: satellite_cloud.jpg에서 구름을 제거하고 건물을 추출해줘.",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        chat_img = gr.Image(label="이미지 업로드 (선택)", type="numpy", height=110)
                with gr.Row():
                    send_btn = gr.Button("전송", variant="primary")
                    reset_btn = gr.Button("대화 초기화", variant="secondary")

                gr.Examples(
                    label="예시 질문",
                    examples=[
                        ["data/images/satellite_cloud.jpg 파일에서 구름을 제거해줘.", None],
                        ["data/images/sar_image.tif에서 선박을 탐지해줘.", None],
                        ["data/images/aerial_photo.jpg에서 건물을 추출해줘.", None],
                        ["data/images/before.jpg와 data/images/after.jpg의 변화를 탐지해줘.", None],
                        ["SAR 원격탐사란 무엇인지 설명해줘.", None],
                        ["Sentinel-2 위성의 특성에 대해 알려줘.", None],
                    ],
                    inputs=[msg, chat_img],
                )
                send_btn.click(chat_fn, [msg, chatbot, chat_img], [chatbot, msg])
                msg.submit(chat_fn, [msg, chatbot, chat_img], [chatbot, msg])
                reset_btn.click(reset_fn, outputs=[chatbot, msg])

            # ━━ 탭 2: 이미지 향상 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🌤️ 이미지 향상"):
                gr.Markdown(
                    "구름 제거, 헤이즈 제거, 초해상도, 노이즈 제거 등 영상 품질 개선 기능입니다."
                )
                with gr.Row():
                    with gr.Column():
                        enh_img_in = gr.Image(
                            label="입력 이미지",
                            type="numpy",
                            value=_load_example("optical"),
                        )
                        enh_mode = gr.Radio(
                            choices=[
                                "구름 제거 (Cloud Removal)",
                                "헤이즈 제거 (Dehazing)",
                                "초해상도 4× (Super Resolution)",
                                "노이즈 제거 (Denoising)",
                            ],
                            value="구름 제거 (Cloud Removal)",
                            label="향상 방법",
                        )
                        enh_btn = gr.Button("실행", variant="primary")
                    with gr.Column():
                        enh_img_out = gr.Image(label="처리 결과", type="numpy")
                        enh_info = gr.Markdown(label="결과 정보")
                enh_btn.click(run_enhancement, [enh_img_in, enh_mode], [enh_img_out, enh_info])

            # ━━ 탭 3: 객체 탐지 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🎯 객체 탐지"):
                gr.Markdown(
                    "수평·회전 바운딩박스 탐지 및 SAR 영상 객체 탐지 기능입니다."
                )
                with gr.Row():
                    with gr.Column():
                        det_img_in = gr.Image(
                            label="입력 이미지",
                            type="numpy",
                            value=_load_example("sar"),
                        )
                        det_type = gr.Radio(
                            choices=[
                                "수평 바운딩박스 (Horizontal)",
                                "회전 바운딩박스 (Rotated/OBB)",
                                "SAR 객체 탐지",
                            ],
                            value="SAR 객체 탐지",
                            label="탐지 방식",
                        )
                        det_cats = gr.Textbox(
                            label="탐지 클래스 (쉼표 구분)",
                            value="vehicle, ship, aircraft",
                            placeholder="vehicle, ship, aircraft",
                        )
                        det_btn = gr.Button("탐지 실행", variant="primary")
                    with gr.Column():
                        det_img_out = gr.Image(label="탐지 결과", type="numpy")
                        det_info = gr.Markdown(label="탐지 정보")
                det_btn.click(run_detection, [det_img_in, det_type, det_cats], [det_img_out, det_info])

            # ━━ 탭 4: 추출·분류 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🏗️ 추출·분류"):
                gr.Markdown(
                    "건물/도로 추출, 장면 분류, 시맨틱 분할 기능입니다."
                )
                with gr.Row():
                    with gr.Column():
                        ext_img_in = gr.Image(
                            label="입력 이미지",
                            type="numpy",
                            value=_load_example("aerial"),
                        )
                        ext_task = gr.Radio(
                            choices=[
                                "건물 추출 (Building Extraction)",
                                "도로 추출 (Road Extraction)",
                                "장면 분류 (Scene Classification)",
                                "시맨틱 분할 (Semantic Segmentation)",
                            ],
                            value="건물 추출 (Building Extraction)",
                            label="분석 작업",
                        )
                        ext_btn = gr.Button("분석 실행", variant="primary")
                    with gr.Column():
                        ext_img_out = gr.Image(label="결과 이미지", type="numpy")
                        ext_img_extra = gr.Image(label="보조 이미지 (분할 맵 등)", type="numpy", visible=True)
                        ext_info = gr.Markdown(label="분석 결과")
                ext_btn.click(
                    run_extraction,
                    [ext_img_in, ext_task],
                    [ext_img_out, ext_img_extra, ext_info],
                )

            # ━━ 탭 5: 변화 탐지·분석 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📊 변화 탐지·분석"):
                gr.Markdown(
                    "변화 탐지, 피해 평가, 토지이용 분류 기능입니다. "
                    "변화 탐지는 Before/After 두 이미지가 필요합니다."
                )
                with gr.Row():
                    with gr.Column():
                        chg_before = gr.Image(
                            label="Before 이미지",
                            type="numpy",
                            value=_load_example("before"),
                        )
                        chg_after = gr.Image(
                            label="After 이미지 (변화 탐지·피해평가)",
                            type="numpy",
                            value=_load_example("after"),
                        )
                        chg_task = gr.Radio(
                            choices=[
                                "변화 탐지 (Change Detection)",
                                "피해 평가 (Damage Assessment)",
                                "토지이용 분류 (Land Use)",
                            ],
                            value="변화 탐지 (Change Detection)",
                            label="분석 작업",
                        )
                        chg_btn = gr.Button("분석 실행", variant="primary")
                    with gr.Column():
                        chg_img_out = gr.Image(label="결과 (Before | Change | After)", type="numpy")
                        chg_map_out = gr.Image(label="변화 맵", type="numpy")
                        chg_info = gr.Markdown(label="분석 결과")
                chg_btn.click(
                    run_change_analysis,
                    [chg_before, chg_after, chg_task],
                    [chg_img_out, chg_map_out, chg_info],
                )

            # ━━ 탭 6: VQA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("❓ 영상 질의응답 (VQA)"):
                gr.Markdown(
                    "원격탐사 영상에 대해 자연어로 질문하고 답변을 받는 VQA 기능입니다."
                )
                with gr.Row():
                    with gr.Column():
                        vqa_img = gr.Image(
                            label="분석 이미지",
                            type="numpy",
                            value=_load_example("optical"),
                        )
                        vqa_question = gr.Textbox(
                            label="질문",
                            placeholder="이 영상에서 보이는 주요 지형지물은 무엇인가요?",
                            lines=2,
                        )
                        vqa_btn = gr.Button("질문 전송", variant="primary")
                        gr.Examples(
                            label="예시 질문",
                            examples=[
                                ["이 SAR 영상에서 선박이 몇 척 보이나요?"],
                                ["구름이 영상의 몇 퍼센트를 덮고 있나요?"],
                                ["이 영상의 토지 피복 유형은 무엇인가요?"],
                                ["건물 밀도가 높은 지역은 어디인가요?"],
                            ],
                            inputs=[vqa_question],
                        )
                    with gr.Column():
                        vqa_answer = gr.Markdown(label="VQA 답변")
                vqa_btn.click(run_vqa, [vqa_img, vqa_question], [vqa_answer])
                vqa_question.submit(run_vqa, [vqa_img, vqa_question], [vqa_answer])

            # ━━ 탭 7: 지식 검색 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🔍 지식 검색 (DualRAG)"):
                gr.Markdown(
                    "원격탐사 도메인 지식 DB를 시맨틱+키워드 이중 검색(DualRAG)으로 조회합니다."
                )
                kb_query = gr.Textbox(
                    label="검색 질문",
                    placeholder="예: SAR 영상에서 선박 탐지 방법은?",
                    lines=2,
                )
                kb_btn = gr.Button("지식 검색", variant="primary")
                kb_result = gr.Markdown(label="검색 결과")
                gr.Examples(
                    label="예시 질문",
                    examples=[
                        ["SAR 원격탐사란 무엇인가요?"],
                        ["Sentinel-2 위성의 밴드 구성은?"],
                        ["구름 제거 알고리즘의 종류는?"],
                        ["NDVI란 무엇이며 어떻게 계산하나요?"],
                        ["초해상도 기법의 원리는?"],
                        ["F-16 전투기의 SAR 영상 특성은?"],
                    ],
                    inputs=[kb_query],
                )
                kb_btn.click(run_knowledge_query, [kb_query], [kb_result])
                kb_query.submit(run_knowledge_query, [kb_query], [kb_result])

            # ━━ 탭 8: 도구 카탈로그 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🔧 도구 카탈로그"):
                tools_desc = agent.get_tool_descriptions()
                categories = {
                    "🌤️ 이미지 향상": [
                        "cloud_removal", "image_dehazing",
                        "super_resolution", "image_denoising",
                    ],
                    "🎯 객체 탐지": [
                        "horizontal_object_detection",
                        "rotated_object_detection",
                        "sar_object_detection",
                    ],
                    "🏗️ 특징 추출": [
                        "building_extraction", "road_extraction",
                    ],
                    "🗺️ 장면 분석": [
                        "scene_classification", "semantic_segmentation",
                        "land_use_classification",
                    ],
                    "📊 변화·분석": [
                        "object_counting", "change_detection", "damage_assessment",
                    ],
                    "✈️ 항공기 분류": [
                        "aircraft_classification_optical",
                        "aircraft_classification_sar",
                    ],
                    "❓ 지식·QA": [
                        "remote_sensing_vqa", "knowledge_query",
                    ],
                }
                md = f"## 📦 RS-Agent 도구 카탈로그 ({len(tools_desc)}개)\n\n"
                for cat, names in categories.items():
                    md += f"### {cat}\n"
                    for name in names:
                        if name in tools_desc:
                            md += f"**`{name}`**  \n{tools_desc[name][:200]}\n\n"
                gr.Markdown(md)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RS-Agent Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  RS-Agent UI starting...")
    print(f"  로컬 LLM 서버가 실행 중인지 확인하세요:")
    print(f"  python scripts/start_local_server.py --backend mock --port 11434")
    print(f"{'='*50}\n")

    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CSS,
    )
