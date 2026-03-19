# RS-Agent: An LLM-driven Remote Sensing Intelligent Agent

Implementation of the paper:
**"RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents"**
([arXiv: 2406.07089](https://arxiv.org/abs/2406.07089))

## Architecture

RS-Agent integrates four key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Central Controller (Claude claude-opus-4-6)              │
│         • Interprets user intent                                │
│         • Plans tool execution strategy                         │
│         • Synthesizes final response                            │
└──────┬──────────────┬──────────────┬───────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐  ┌───────────────┐  ┌────────────────┐
│ Dynamic  │  │ Solution Space│  │ Knowledge Space │
│ Toolkit  │  │(Task-Aware    │  │  (DualRAG)      │
│          │  │ Retrieval)    │  │                 │
│ 19 Tools │  │ 16 Solutions  │  │ Domain Knowledge│
└──────────┘  └───────────────┘  └────────────────┘
```

### 1. Central Controller
- **Model**: `claude-opus-4-6` with adaptive thinking
- Orchestrates the entire agent pipeline
- Manages multi-turn conversations

### 2. Dynamic Toolkit (19 Tools)

| Category | Tools |
|----------|-------|
| Image Enhancement | Cloud Removal, Dehazing, Super-Resolution, Denoising |
| Object Detection | Horizontal Detection, Rotated Detection, SAR Detection |
| Scene Analysis | Scene Classification, Semantic Segmentation |
| Feature Extraction | Building Extraction, Road Extraction |
| Specialized | Aircraft Classification (Optical/SAR), Damage Assessment |
| Counting & Change | Object Counting, Change Detection, Land Use Classification |
| Knowledge & Q&A | Remote Sensing VQA, Knowledge Query |

### 3. Solution Space (Task-Aware Retrieval)
- Stores expert solution templates for common RS tasks
- Uses semantic similarity to retrieve relevant solutions
- Guides the LLM with expert tool selection strategies

### 4. Knowledge Space (DualRAG)
- Domain-specific knowledge database (aircraft, sensors, techniques)
- **DualRAG**: Combines semantic (dense) + keyword-weighted BM25 (sparse) retrieval
- Weighted fusion of both retrieval paths for optimal accuracy

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd RS-Agent

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Usage

### Interactive CLI

```bash
python main.py
```

### Single Query

```bash
python main.py --query "Detect vehicles in /path/to/image.jpg"
```

### Web UI (Gradio)

```bash
python main.py --ui
# or
python app.py
```

### Python API

```python
from rs_agent import RSAgent
from rs_agent.utils import load_config

config = load_config("config.yaml")
agent = RSAgent(config=config)

# Chat with the agent
response = agent.chat("What aircraft types are visible in this satellite image?")
print(response)

# Query knowledge base directly (DualRAG)
knowledge = agent.knowledge_query("SAR remote sensing principles")
print(knowledge["answer"])
```

## Configuration

Edit `config.yaml` to customize:
- LLM model and parameters
- Solution Space retrieval settings
- Knowledge Space DualRAG weights
- Toolkit options

## Novel Contributions

### Task-Aware Retrieval
Expert solution templates are retrieved based on semantic similarity to the user's query.
The retrieved solutions guide the LLM's tool selection and planning.

### DualRAG
A dual-path retrieval system combining:
1. **Semantic Path**: Dense vector similarity using sentence transformers
2. **Keyword Path**: Weighted BM25 with domain-specific term boosting

Scores are combined: `combined = α × semantic + β × keyword_bm25`

## Performance (from paper)
- >95% task planning accuracy
- Supports 18+ remote sensing tasks across 9 datasets
- Compatible with open-source and proprietary LLMs

## Citation

```bibtex
@article{xu2024rsagent,
  title={RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents},
  author={Xu, Wenjia and Yu, Zijian and Wang, Yixu and Wang, Jiuniu and Peng, Mugen},
  journal={arXiv preprint arXiv:2406.07089},
  year={2024}
}
```
