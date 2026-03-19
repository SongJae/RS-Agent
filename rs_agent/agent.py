"""
RS-Agent Central Controller.

Implements the LLM-based Central Controller that:
1. Interprets user queries via natural language understanding
2. Retrieves expert guidance from the Solution Space (Task-Aware Retrieval)
3. Retrieves domain knowledge from the Knowledge Space (DualRAG)
4. Selects and executes appropriate tools from the Dynamic Toolkit
5. Produces professional remote sensing analysis results

Architecture follows the paper:
    "RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents"
    arXiv: 2406.07089

LLM Backend 지원:
    - "anthropic"         : Anthropic Claude API (외부망 필요)
    - "openai_compatible" : OpenAI 호환 로컬 서버 (vLLM, LM Studio, Ollama /v1 등)
    - "ollama"            : Ollama 네이티브 API
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator

from rs_agent.toolkit.rs_tools import get_all_tools, TOOL_REGISTRY
from rs_agent.solution_space import SolutionDatabase, TaskAwareRetrieval
from rs_agent.knowledge_space import KnowledgeDatabase, DualRAG

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are RS-Agent, an expert remote sensing intelligent agent powered by advanced AI.
You have access to 19 specialized remote sensing tools and comprehensive domain knowledge.

Your capabilities include:
- Image Enhancement: cloud removal, dehazing, super-resolution, denoising
- Object Detection: horizontal/rotated detection, SAR object detection
- Scene Analysis: scene classification, semantic segmentation
- Feature Extraction: building extraction, road extraction
- Specialized Analysis: aircraft classification (optical/SAR), damage assessment
- Counting & Change: object counting, change detection, land use classification
- Knowledge Q&A: remote sensing VQA, domain knowledge queries

When handling user requests:
1. Analyze the user's task and select the most appropriate tools
2. Follow expert solution templates when available
3. Use domain knowledge to enhance your analysis
4. Explain your tool selection and analysis process
5. Provide professional, accurate results

Always be professional, precise, and helpful. If a task requires multiple tools,
plan and execute them in the correct sequence."""


class RSAgent:
    """
    RS-Agent: LLM-driven Remote Sensing Intelligent Agent.

    Integrates four key components:
    1. Central Controller (LLM — Anthropic 또는 로컬 LLM)
    2. Dynamic Toolkit (19 remote sensing tools)
    3. Solution Space (Task-Aware Retrieval)
    4. Knowledge Space (DualRAG)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_client()
        self._setup_knowledge_space()
        self._setup_solution_space()
        self._setup_toolkit()
        self.conversation_history: List[Dict] = []

    # ──────────────────────────────────────────────
    # Client Setup
    # ──────────────────────────────────────────────

    def _setup_client(self):
        """
        LLM 백엔드를 초기화합니다.

        config.yaml의 llm.backend 값에 따라 분기:
          - "anthropic"         → Anthropic Python SDK
          - "openai_compatible" → OpenAI Python SDK (base_url 변경)
          - "ollama"            → OpenAI Python SDK (Ollama /v1 엔드포인트)
        """
        llm_cfg = self.config.get("llm", {})
        self.backend = llm_cfg.get("backend", "anthropic")
        self.model = llm_cfg.get("model", "claude-opus-4-6")
        self.max_tokens = llm_cfg.get("max_tokens", 4096)

        if self.backend == "anthropic":
            self._setup_anthropic(llm_cfg)
        elif self.backend in ("openai_compatible", "ollama"):
            self._setup_openai_compatible(llm_cfg)
        else:
            raise ValueError(
                f"Unknown LLM backend: '{self.backend}'. "
                "Choose from: anthropic, openai_compatible, ollama"
            )

        logger.info(
            f"Central Controller initialized: backend={self.backend}, model={self.model}"
        )

    def _setup_anthropic(self, llm_cfg: Dict):
        """Anthropic Claude API 클라이언트 초기화."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY") or llm_cfg.get("api_key", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY 환경변수 또는 config의 llm.api_key가 필요합니다."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.thinking_config = llm_cfg.get("thinking", {"type": "adaptive"})

    def _setup_openai_compatible(self, llm_cfg: Dict):
        """OpenAI 호환 로컬 LLM 클라이언트 초기화 (vLLM / Ollama / LM Studio)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")

        # 환경변수 우선, 없으면 config, 없으면 기본값
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or llm_cfg.get("base_url", "http://localhost:11434/v1")
        )
        api_key = (
            os.environ.get("LLM_API_KEY")
            or llm_cfg.get("api_key", "ollama")
        )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"OpenAI-compatible client → {base_url}")

    # ──────────────────────────────────────────────
    # Knowledge / Solution / Toolkit Setup
    # ──────────────────────────────────────────────

    def _setup_knowledge_space(self):
        """Initialize the Knowledge Space with DualRAG."""
        ks_config = self.config.get("knowledge_space", {})
        data_dir = ks_config.get("data_dir", "data/knowledge")
        self.knowledge_db = KnowledgeDatabase(data_dir=data_dir)
        self.dual_rag = DualRAG(
            knowledge_db=self.knowledge_db,
            embedding_model=ks_config.get("embedding_model", "all-MiniLM-L6-v2"),
            local_model_path=ks_config.get("local_model_path", ""),
            top_k_semantic=ks_config.get("top_k_semantic", 5),
            top_k_keyword=ks_config.get("top_k_keyword", 5),
            semantic_weight=ks_config.get("semantic_weight", 0.6),
            keyword_weight=ks_config.get("keyword_weight", 0.4),
        )
        logger.info(f"Knowledge Space initialized: {len(self.knowledge_db)} documents")

    def _setup_solution_space(self):
        """Initialize the Solution Space with Task-Aware Retrieval."""
        ss_config = self.config.get("solution_space", {})
        data_dir = ss_config.get("data_dir", "data/solutions")
        self.solution_db = SolutionDatabase(data_dir=data_dir)
        self.task_retrieval = TaskAwareRetrieval(
            solution_db=self.solution_db,
            embedding_model=ss_config.get("embedding_model", "all-MiniLM-L6-v2"),
            local_model_path=ss_config.get("local_model_path", ""),
            top_k=ss_config.get("top_k", 3),
            similarity_threshold=ss_config.get("similarity_threshold", 0.3),
        )
        logger.info(f"Solution Space initialized: {len(self.solution_db)} solutions")

    def _setup_toolkit(self):
        """Initialize the Dynamic Toolkit with all 19 RS tools."""
        self.tools = get_all_tools(knowledge_space=self.dual_rag)
        self.tool_map = {tool.name: tool for tool in self.tools}
        # 백엔드에 따라 다른 tool 포맷 사전 생성
        if self.backend == "anthropic":
            self._tools_formatted = [tool.to_anthropic_tool() for tool in self.tools]
        else:
            self._tools_formatted = [tool.to_openai_tool() for tool in self.tools]
        logger.info(f"Dynamic Toolkit initialized: {len(self.tools)} tools")

    # ──────────────────────────────────────────────
    # System Prompt
    # ──────────────────────────────────────────────

    def _build_system_prompt(self, query: str) -> str:
        """Build system prompt enriched with Task-Aware Retrieval context."""
        base_prompt = self.config.get("agent", {}).get("system_prompt", SYSTEM_PROMPT)
        relevant_solutions = self.task_retrieval.retrieve(query)
        solution_context = self.task_retrieval.format_as_context(relevant_solutions)
        if solution_context:
            return f"{base_prompt}\n\n{solution_context}"
        return base_prompt

    # ──────────────────────────────────────────────
    # Tool Execution
    # ──────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool and return the result as a JSON string."""
        if tool_name not in self.tool_map:
            return json.dumps(
                {"status": "error", "message": f"Tool '{tool_name}' not found."}
            )
        tool = self.tool_map[tool_name]
        try:
            result = tool.run(**tool_input)
            if result.success:
                output = {
                    "status": "success",
                    "message": result.message,
                    "data": result.output,
                }
            else:
                output = {"status": "error", "message": result.message}
            return json.dumps(output, indent=2, default=str)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    # ──────────────────────────────────────────────
    # Chat — Anthropic backend
    # ──────────────────────────────────────────────

    def _chat_anthropic(self, messages: List[Dict], system_prompt: str) -> str:
        """Agentic loop for Anthropic Claude backend."""
        max_iterations = self.config.get("agent", {}).get("max_iterations", 10)

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking=self.thinking_config,
                system=system_prompt,
                tools=self._tools_formatted,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            if response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for tc in tool_calls:
                    logger.info(f"Executing tool: {tc.name}")
                    result = self._execute_tool(tc.name, tc.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

        return "Maximum iterations reached. Please refine your query."

    # ──────────────────────────────────────────────
    # Chat — OpenAI-compatible backend
    # ──────────────────────────────────────────────

    def _chat_openai_compatible(self, messages: List[Dict], system_prompt: str) -> str:
        """Agentic loop for OpenAI-compatible local LLM backend."""
        max_iterations = self.config.get("agent", {}).get("max_iterations", 10)
        # OpenAI 포맷: system 메시지를 messages 앞에 삽입
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=self._tools_formatted,
                tool_choice="auto",
                messages=full_messages,
            )

            choice = response.choices[0]
            msg = choice.message

            # 도구 호출 없음 → 최종 답변
            if not msg.tool_calls:
                return msg.content or ""

            # 도구 호출 존재 → assistant 메시지 추가
            full_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 각 도구 실행 후 결과를 tool 메시지로 추가
            for tc in msg.tool_calls:
                logger.info(f"Executing tool: {tc.function.name}")
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                result = self._execute_tool(tc.function.name, tool_input)
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Maximum iterations reached. Please refine your query."

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the full RS-Agent pipeline.

        Pipeline:
        1. Task-Aware Retrieval → expert solution context
        2. LLM Central Controller with tool-use loop
        3. Tool execution and result synthesis
        """
        logger.info(f"Processing query: {user_message[:100]}...")
        system_prompt = self._build_system_prompt(user_message)
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = list(self.conversation_history)

        if self.backend == "anthropic":
            response_text = self._chat_anthropic(messages, system_prompt)
        else:
            response_text = self._chat_openai_compatible(messages, system_prompt)

        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Process a user message with streaming output.

        Anthropic 백엔드에서는 스트리밍을 지원합니다.
        OpenAI 호환 백엔드에서는 non-streaming으로 fallback합니다.
        """
        if self.backend == "anthropic":
            yield from self._stream_chat_anthropic(user_message)
        else:
            response = self.chat(user_message)
            yield response

    def _stream_chat_anthropic(self, user_message: str) -> Generator[str, None, None]:
        """Anthropic 백엔드 스트리밍 구현."""
        system_prompt = self._build_system_prompt(user_message)
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = list(self.conversation_history)
        max_iterations = self.config.get("agent", {}).get("max_iterations", 10)
        full_response = ""

        for _ in range(max_iterations):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking=self.thinking_config,
                system=system_prompt,
                tools=self._tools_formatted,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            full_response += chunk
                            yield chunk

                final_msg = stream.get_final_message()

                if final_msg.stop_reason == "end_turn":
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": full_response,
                    })
                    return

                if final_msg.stop_reason == "tool_use":
                    tool_calls = [b for b in final_msg.content if b.type == "tool_use"]
                    messages.append({"role": "assistant", "content": final_msg.content})

                    tool_results = []
                    for tc in tool_calls:
                        yield f"\n\n[Executing: {tc.name}...]\n"
                        result = self._execute_tool(tc.name, tc.input)
                        yield f"[Tool Result]\n```json\n{result}\n```\n\n"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result,
                        })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

    def knowledge_query(self, query: str) -> Dict[str, Any]:
        """Directly query the Knowledge Space using DualRAG."""
        return self.dual_rag.query(query)

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def get_available_tools(self) -> List[str]:
        return list(self.tool_map.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        return {tool.name: tool.description for tool in self.tools}

    def __repr__(self) -> str:
        return (
            f"RSAgent(backend={self.backend!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, "
            f"solutions={len(self.solution_db)}, "
            f"knowledge={len(self.knowledge_db)})"
        )
