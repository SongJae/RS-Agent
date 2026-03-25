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
│                   Central Controller (LLM)                      │
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
- 외부망: Anthropic Claude API
- **폐쇄망**: 내장 로컬 서버 (Mock / Transformers 백엔드)

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

---

## 설치 방법

### A. 외부망 환경 (인터넷 연결 가능)

```bash
git clone <repo-url>
cd RS-Agent
pip install -r requirements.txt
```

### B. 폐쇄망 환경 (인터넷 연결 불가) — Miniconda 패키지 이식

폐쇄망에서는 pip/conda로 패키지를 설치할 수 없습니다.
**인터넷 연결 PC에서 환경 전체를 미리 패킹한 뒤, USB/HDD로 반입**합니다.

#### 1단계: 인터넷 PC에서 패키지 묶기

[Miniconda 설치](https://docs.conda.io/en/latest/miniconda.html) 후:

```bash
# GPU 환경 (A100, CUDA 12.1)
bash scripts/pack_conda_env.sh

# CPU-only 환경
bash scripts/pack_conda_env.sh --cpu-only

# Flash Attention 2 제외 (빌드 생략)
bash scripts/pack_conda_env.sh --no-flash-attn
```

생성 결과:
```
rs-agent-env.tar.gz   ← conda 환경 전체 (~4–8 GB)
```

#### 2단계: USB/HDD에 복사

```
/usb/
├── rs-agent-env.tar.gz      ← pack_conda_env.sh 생성물
└── RS-Agent/                ← 이 저장소 (코드 + 모델 + 데이터 포함)
```

> `RS-Agent/models/all-MiniLM-L6-v2/` — 임베딩 모델이 이미 포함되어 있습니다.
> 실제 LLM(Qwen2.5-7B 등)을 사용하려면 별도로 `RS-Agent/models/` 아래에 복사하세요.

#### 3단계: 폐쇄망 PC에서 설치

```bash
# RS-Agent/와 rs-agent-env.tar.gz가 같은 디렉토리에 있는 경우
bash RS-Agent/scripts/install_offline.sh

# tar.gz 경로를 직접 지정하는 경우
bash RS-Agent/scripts/install_offline.sh /media/usb/rs-agent-env.tar.gz
```

설치 후 다음 실행 스크립트가 자동 생성됩니다:
- `run_server.sh` — LLM 서버 (Mock 모드)
- `run_server_gpu.sh` — LLM 서버 (GPU 모드)
- `run_ui.sh` — Gradio Web UI

---

## 실행 방법

### 외부망: Anthropic Claude API

```bash
cp .env.example .env
# .env에 ANTHROPIC_API_KEY 입력 후:
python app.py
```

### 폐쇄망: 로컬 서버 + Web UI

터미널을 2개 열어 실행합니다.

**터미널 1 — LLM 서버 시작:**

```bash
# Mock 모드 (GPU 불필요, 즉시 실행)
bash run_server.sh

# A100 GPU — FP16 (~24 GB VRAM)
bash run_server_gpu.sh models/Qwen2.5-7B-Instruct

# A100 GPU — 4-bit 양자화 (~12 GB VRAM)
bash run_server_gpu.sh models/Qwen2.5-7B-Instruct --load-in-4bit
```

**터미널 2 — Web UI 시작:**

```bash
bash run_ui.sh
# 브라우저: http://localhost:7860
```

### CLI 사용

```bash
# 환경 활성화 후
source ~/rs-agent-env/bin/activate

python main.py
python main.py --query "Detect vehicles in data/images/aerial_photo.jpg"
```

### Python API

```python
from rs_agent import RSAgent
from rs_agent.utils import load_config

config = load_config("config.yaml")
agent = RSAgent(config=config)

response = agent.chat("이 위성 이미지에서 건물을 추출해줘")
print(response)

knowledge = agent.knowledge_query("SAR 원격탐사 원리")
print(knowledge["answer"])
```

---

## 구성 파일

`config.yaml`에서 다음을 설정합니다:

| 항목 | 설명 | 기본값 |
|------|------|--------|
| `llm.backend` | `anthropic` / `openai_compatible` | `openai_compatible` |
| `llm.base_url` | 로컬 서버 주소 | `http://localhost:11434/v1` |
| `llm.model` | 모델 이름 | `mock-rs-agent` |
| `knowledge_space.local_model_path` | 임베딩 모델 경로 | `models/all-MiniLM-L6-v2` |
| `solution_space.local_model_path` | 임베딩 모델 경로 | `models/all-MiniLM-L6-v2` |

---

## 폐쇄망 구성 요소 체크리스트

| 구성 요소 | 파일 위치 | 비고 |
|-----------|----------|------|
| 실행 환경 | `rs-agent-env.tar.gz` | `pack_conda_env.sh`로 생성 |
| 임베딩 모델 | `models/all-MiniLM-L6-v2/` | 저장소에 포함됨 (87 MB) |
| LLM (선택) | `models/<모델명>/` | 별도 반입, Mock 모드는 불필요 |
| 지식 데이터 | `data/knowledge/*.json` | 저장소에 포함됨 |
| 솔루션 데이터 | `data/solutions/*.json` | 저장소에 포함됨 |
| 예시 이미지 | `data/images/` | 저장소에 포함됨 |

---

## Novel Contributions

### Task-Aware Retrieval
Expert solution templates are retrieved based on semantic similarity to the user's query.
The retrieved solutions guide the LLM's tool selection and planning.

### DualRAG
A dual-path retrieval system combining:
1. **Semantic Path**: Dense vector similarity using sentence transformers
2. **Keyword Path**: Weighted BM25 with domain-specific term boosting

Scores are combined: `combined = α × semantic + β × keyword_bm25`

---

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
