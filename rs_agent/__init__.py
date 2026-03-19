"""
RS-Agent: An LLM-driven Remote Sensing Intelligent Agent

Based on the paper: "RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents"
(arXiv: 2406.07089)

Architecture:
    1. Central Controller (LLM) - claude-opus-4-6
    2. Dynamic Toolkit - 19 specialized remote sensing tools
    3. Solution Space - Task-Aware Retrieval for expert guidance
    4. Knowledge Space - DualRAG for domain-specific knowledge
"""

from rs_agent.agent import RSAgent

__version__ = "1.0.0"
__all__ = ["RSAgent"]
