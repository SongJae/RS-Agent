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
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator

import anthropic

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
    1. Central Controller (Claude claude-opus-4-6 with adaptive thinking)
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

    def _setup_client(self):
        """Initialize the Anthropic client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = self.config.get("llm", {}).get("model", "claude-opus-4-6")
        self.max_tokens = self.config.get("llm", {}).get("max_tokens", 4096)
        logger.info(f"Central Controller initialized: {self.model}")

    def _setup_knowledge_space(self):
        """Initialize the Knowledge Space with DualRAG."""
        ks_config = self.config.get("knowledge_space", {})
        data_dir = ks_config.get("data_dir", "data/knowledge")
        self.knowledge_db = KnowledgeDatabase(data_dir=data_dir)
        self.dual_rag = DualRAG(
            knowledge_db=self.knowledge_db,
            embedding_model=ks_config.get("embedding_model", "all-MiniLM-L6-v2"),
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
            top_k=ss_config.get("top_k", 3),
            similarity_threshold=ss_config.get("similarity_threshold", 0.3),
        )
        logger.info(f"Solution Space initialized: {len(self.solution_db)} solutions")

    def _setup_toolkit(self):
        """Initialize the Dynamic Toolkit with all 19 RS tools."""
        self.tools = get_all_tools(knowledge_space=self.dual_rag)
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.anthropic_tools = [tool.to_anthropic_tool() for tool in self.tools]
        logger.info(f"Dynamic Toolkit initialized: {len(self.tools)} tools")

    def _build_system_prompt(self, query: str) -> str:
        """
        Build the system prompt enriched with Task-Aware Retrieval context.

        This is the core of the Solution Space integration:
        expert solutions are retrieved and injected into the system context.
        """
        base_prompt = self.config.get("agent", {}).get("system_prompt", SYSTEM_PROMPT)

        # Task-Aware Retrieval: get relevant expert solutions
        relevant_solutions = self.task_retrieval.retrieve(query)
        solution_context = self.task_retrieval.format_as_context(relevant_solutions)

        if solution_context:
            return f"{base_prompt}\n\n{solution_context}"
        return base_prompt

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool and return the result as a string."""
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found in toolkit."

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
                output = {
                    "status": "error",
                    "message": result.message,
                }
            return json.dumps(output, indent=2, default=str)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the full RS-Agent pipeline.

        Pipeline:
        1. Task-Aware Retrieval → expert solution context
        2. DualRAG knowledge enrichment (if knowledge query detected)
        3. LLM Central Controller with tool use loop
        4. Tool execution and result synthesis
        """
        logger.info(f"Processing query: {user_message[:100]}...")

        # Build enriched system prompt via Task-Aware Retrieval
        system_prompt = self._build_system_prompt(user_message)

        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Agent loop: LLM → tool calls → results → LLM
        messages = list(self.conversation_history)
        max_iterations = self.config.get("agent", {}).get("max_iterations", 10)

        for iteration in range(max_iterations):
            logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

            # Call the Central Controller (LLM)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=system_prompt,
                tools=self.anthropic_tools,
                messages=messages,
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text,
                })
                return final_text

            elif response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                thinking_blocks = [b for b in response.content if b.type == "thinking"]

                if self.config.get("agent", {}).get("verbose", True):
                    for tb in thinking_blocks:
                        logger.debug(f"[Thinking] {tb.thinking[:200]}...")

                # Append assistant message (with tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                # Execute tools and collect results
                tool_results = []
                for tool_call in tool_calls:
                    logger.info(f"Executing tool: {tool_call.name} with {tool_call.input}")
                    result = self._execute_tool(tool_call.name, tool_call.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                    })

                # Append tool results as user message
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

        return "Maximum iterations reached. Please refine your query."

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Process a user message with streaming output.

        Yields text chunks as they are generated by the Central Controller.
        Tool execution results are yielded as formatted messages.
        """
        system_prompt = self._build_system_prompt(user_message)
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = list(self.conversation_history)
        max_iterations = self.config.get("agent", {}).get("max_iterations", 10)
        full_response = ""

        for iteration in range(max_iterations):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=system_prompt,
                tools=self.anthropic_tools,
                messages=messages,
            ) as stream:
                current_tool_calls = []
                current_text = ""

                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            current_text += chunk
                            full_response += chunk
                            yield chunk

                final_msg = stream.get_final_message()

                if final_msg.stop_reason == "end_turn":
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": full_response,
                    })
                    return

                elif final_msg.stop_reason == "tool_use":
                    tool_calls = [b for b in final_msg.content if b.type == "tool_use"]
                    messages.append({"role": "assistant", "content": final_msg.content})

                    tool_results = []
                    for tool_call in tool_calls:
                        yield f"\n\n[Executing: {tool_call.name}...]\n"
                        result = self._execute_tool(tool_call.name, tool_call.input)
                        yield f"[Tool Result]\n```json\n{result}\n```\n\n"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result,
                        })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

    def knowledge_query(self, query: str) -> Dict[str, Any]:
        """
        Directly query the Knowledge Space using DualRAG.

        Returns the retrieved knowledge without going through the agent loop.
        """
        return self.dual_rag.query(query)

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def get_available_tools(self) -> List[str]:
        """Return list of available tool names."""
        return list(self.tool_map.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return tool name → description mapping."""
        return {tool.name: tool.description for tool in self.tools}

    def __repr__(self) -> str:
        return (
            f"RSAgent(model={self.model!r}, "
            f"tools={len(self.tools)}, "
            f"solutions={len(self.solution_db)}, "
            f"knowledge={len(self.knowledge_db)})"
        )
