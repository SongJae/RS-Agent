"""
RS-Agent Gradio Web UI.

Provides a user-friendly web interface for interacting with RS-Agent.
"""

import os
import json
import gradio as gr
from typing import List, Tuple


def create_gradio_app(agent=None):
    """Create and return the Gradio application."""

    if agent is None:
        from dotenv import load_dotenv
        load_dotenv()
        from rs_agent import RSAgent
        from rs_agent.utils import load_config
        try:
            config = load_config("config.yaml")
        except FileNotFoundError:
            config = {}
        agent = RSAgent(config=config)

    def chat_fn(
        message: str,
        history: List[Tuple[str, str]],
        image_path: str,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Handle chat messages."""
        if not message.strip():
            return history, ""

        # If image provided, prepend path info to message
        if image_path:
            message = f"[Image: {image_path}]\n{message}"

        try:
            response = agent.chat(message)
        except Exception as e:
            response = f"Error: {str(e)}"

        history.append((message, response))
        return history, ""

    def knowledge_query_fn(query: str) -> str:
        """Query the knowledge base directly."""
        if not query.strip():
            return "Please enter a query."
        try:
            result = agent.knowledge_query(query)
            output = f"**Query:** {result['query']}\n\n"
            output += f"**Answer:**\n{result['answer']}\n\n"
            if result.get("sources"):
                output += "**Sources:**\n"
                for src in result["sources"][:3]:
                    output += f"- [{src['title']}] (score: {src['score']:.3f})\n"
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def reset_chat():
        """Reset the conversation."""
        agent.reset_conversation()
        return [], ""

    def get_tools_info() -> str:
        """Get formatted tool descriptions."""
        tools = agent.get_tool_descriptions()
        output = f"## RS-Agent Toolkit ({len(tools)} Tools)\n\n"
        categories = {
            "Image Enhancement": ["cloud_removal", "image_dehazing", "super_resolution", "image_denoising"],
            "Object Detection": ["horizontal_object_detection", "rotated_object_detection", "sar_object_detection"],
            "Scene Analysis": ["scene_classification", "semantic_segmentation"],
            "Feature Extraction": ["building_extraction", "road_extraction"],
            "Specialized Analysis": ["aircraft_classification_optical", "aircraft_classification_sar", "damage_assessment"],
            "Counting & Change": ["object_counting", "change_detection", "land_use_classification"],
            "Knowledge & Q&A": ["remote_sensing_vqa", "knowledge_query"],
        }
        for category, tool_names in categories.items():
            output += f"### {category}\n"
            for name in tool_names:
                if name in tools:
                    output += f"**`{name}`**: {tools[name][:120]}...\n\n"
        return output

    # ── Gradio UI Layout ──────────────────────────────────────────────────────

    with gr.Blocks(
        title="RS-Agent: Remote Sensing Intelligent Agent",
        theme=gr.themes.Soft(),
        css="""
        .chatbot { height: 500px; }
        .tool-info { font-size: 0.9em; }
        """,
    ) as demo:

        gr.Markdown("""
        # 🛰️ RS-Agent: Remote Sensing Intelligent Agent
        **Automating Remote Sensing Tasks through Intelligent Agents** | [arXiv: 2406.07089](https://arxiv.org/abs/2406.07089)

        RS-Agent integrates:
        - 🧠 **Central Controller** (Claude claude-opus-4-6 with adaptive thinking)
        - 🔧 **19 Specialized Remote Sensing Tools**
        - 📚 **Solution Space** (Task-Aware Retrieval)
        - 🔍 **Knowledge Space** (DualRAG)
        """)

        with gr.Tabs():
            # ── Chat Tab ──────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    label="RS-Agent",
                    elem_classes="chatbot",
                    show_label=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        msg_input = gr.Textbox(
                            label="Your Message",
                            placeholder=(
                                "Ask RS-Agent to analyze images or answer remote sensing questions...\n"
                                "Example: 'Detect vehicles in /path/to/image.jpg'"
                            ),
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        image_input = gr.Textbox(
                            label="Image Path (optional)",
                            placeholder="/path/to/image.jpg",
                        )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("Reset Conversation", variant="secondary")

                gr.Examples(
                    examples=[
                        ["Detect and count all vehicles in the image", ""],
                        ["What type of scene is shown in this remote sensing image?", ""],
                        ["Extract building footprints and calculate coverage ratio", ""],
                        ["Compare these two images and detect land use changes", ""],
                        ["Classify the aircraft types visible at this airbase", ""],
                        ["Assess damage in the post-earthquake satellite image", ""],
                        ["What are the characteristics of SAR remote sensing?", ""],
                        ["Tell me about the F-16 recognition features in satellite imagery", ""],
                    ],
                    inputs=[msg_input, image_input],
                )

                submit_btn.click(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot, image_input],
                    outputs=[chatbot, msg_input],
                )
                msg_input.submit(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot, image_input],
                    outputs=[chatbot, msg_input],
                )
                reset_btn.click(
                    fn=reset_chat,
                    outputs=[chatbot, msg_input],
                )

            # ── Knowledge Query Tab ────────────────────────────────────────────
            with gr.Tab("🔍 Knowledge Query (DualRAG)"):
                gr.Markdown("""
                ### Direct Knowledge Base Query via DualRAG
                Query the remote sensing knowledge base using the dual-path retrieval system
                (semantic + keyword-weighted BM25).
                """)

                kb_query = gr.Textbox(
                    label="Knowledge Query",
                    placeholder="e.g., 'C-130 transport aircraft recognition features'",
                )
                kb_result = gr.Markdown(label="Retrieved Knowledge")
                kb_btn = gr.Button("Query Knowledge Base", variant="primary")

                gr.Examples(
                    examples=[
                        ["F-16 fighter jet recognition features in SAR imagery"],
                        ["How does Sentinel-1 SAR work?"],
                        ["Methods for cloud removal in optical remote sensing"],
                        ["NDVI vegetation index calculation"],
                        ["Change detection algorithms for urban monitoring"],
                    ],
                    inputs=[kb_query],
                )

                kb_btn.click(fn=knowledge_query_fn, inputs=[kb_query], outputs=[kb_result])
                kb_query.submit(fn=knowledge_query_fn, inputs=[kb_query], outputs=[kb_result])

            # ── Tools Info Tab ────────────────────────────────────────────────
            with gr.Tab("🔧 Available Tools"):
                gr.Markdown(get_tools_info(), elem_classes="tool-info")

            # ── About Tab ─────────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                ## RS-Agent Architecture

                **Based on:** "RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents"
                (arXiv: 2406.07089, Xu et al., 2024)

                ### Four Key Components

                #### 1. 🧠 Central Controller
                - Model: `{agent.model}`
                - Adaptive thinking enabled for complex reasoning
                - Multi-turn conversation support

                #### 2. 🔧 Dynamic Toolkit
                - **{len(agent.tools)} specialized remote sensing tools**
                - Image enhancement: cloud removal, dehazing, super-resolution, denoising
                - Detection: horizontal/rotated/SAR object detection
                - Analysis: scene classification, segmentation, building/road extraction
                - Specialized: aircraft classification, damage assessment, VQA

                #### 3. 📚 Solution Space (Task-Aware Retrieval)
                - **{len(agent.solution_db)} expert solution templates**
                - Semantic retrieval using sentence transformers
                - Guides tool selection and task decomposition

                #### 4. 🔍 Knowledge Space (DualRAG)
                - **{len(agent.knowledge_db)} knowledge documents**
                - Dual-path retrieval: semantic + keyword-weighted BM25
                - Domain knowledge: aircraft specs, sensors, RS techniques

                ### Novel Contributions
                - **Task-Aware Retrieval**: Expert-guided tool selection
                - **DualRAG**: Weighted dual-path knowledge retrieval
                """)

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
