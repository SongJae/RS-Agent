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

### 4. Knowledge Space (DualRAG)
- Domain-specific knowledge database (aircraft, sensors, techniques)
- **DualRAG**: Combines semantic (dense) + keyword-weighted BM25 (sparse) retrieval

---

## 저장소 구조

```
RS-Agent/
├── install/
│   └── Miniconda3-latest-Linux-x86_64.sh  ← Miniconda 설치 파일 (155 MB)
├── packages/
│   └── wheels/                             ← 코어 pip 패키지 (104 MB, 인터넷 불필요)
│       ├── numpy-*.whl
│       ├── faiss_cpu-*.whl
│       └── ...
├── models/
│   └── all-MiniLM-L6-v2/                  ← 임베딩 모델 (87 MB, 포함됨)
├── data/
│   ├── images/                             ← 예시 위성 이미지
│   ├── knowledge/                          ← 도메인 지식 DB (JSON)
│   └── solutions/                          ← 솔루션 템플릿 (JSON)
├── rs_agent/                               ← 에이전트 코드
├── scripts/
│   ├── pack_conda_env.sh                   ← [인터넷 PC] 환경 패킹
│   ├── install_offline.sh                  ← [폐쇄망 PC] 환경 설치
│   └── start_local_server.py              ← LLM 서버 실행
├── app.py                                  ← Gradio Web UI
└── rs-agent-env.tar.gz                     ← conda 환경 전체 (3 GB, git 제외)
                                              pack_conda_env.sh 실행 후 생성
```

> **`rs-agent-env.tar.gz`는 git에 포함되지 않습니다 (3 GB).**
> `pack_conda_env.sh`로 생성한 뒤 USB에 함께 복사하세요.

---

## 설치 방법

### A. 외부망 환경

```bash
git clone <repo-url>
cd RS-Agent
pip install -r requirements.txt
```

---

### B. 폐쇄망 환경 — 단계별 가이드

> **ML 패키지(~2.8 GB)는 git에 포함되지 않습니다.**
> SFTP 또는 USB로 전송해야 합니다. 아래 두 가지 방법 중 하나를 선택하세요.

---

#### 방법 1: SFTP 전송 (권장 — USB 없이 네트워크로 전송)

```
[인터넷 PC]                              [폐쇄망 PC]
   ①  git clone + download_ml.sh
   ②  pack_conda_env.sh → rs-agent-env.tar.gz
   ③  sftp 전송 ──────────────────────────────► ④  install_offline.sh
```

**① 인터넷 PC — ML 패키지 다운로드**

```bash
git clone <repo-url>
cd RS-Agent

# ML wheels 다운로드 → packages-ml.tar.gz 생성 (~2.8 GB)
bash scripts/download_ml.sh
```

**② 인터넷 PC — 환경 패킹**

```bash
# GPU 환경 (CUDA) — packages-ml.tar.gz 가 있으면 자동 인식
bash scripts/pack_conda_env.sh
# → rs-agent-env.tar.gz (~3 GB) 생성

# Flash Attention 빌드 시간을 줄이려면
bash scripts/pack_conda_env.sh --no-flash-attn
```

**③ SFTP로 폐쇄망 PC에 전송**

```bash
# 인터넷 PC에서 실행 (사용자 계정/IP는 실제 환경에 맞게 변경)
sftp user@<폐쇄망-IP>
sftp> put rs-agent-env.tar.gz
sftp> exit
```

**④ 폐쇄망 PC — 설치**

```bash
# git repo는 별도로 git clone 또는 zip 복사
git clone <repo-url>   # 또는 zip 압축 해제

bash RS-Agent/scripts/install_offline.sh ~/rs-agent-env.tar.gz
```

---

#### 방법 2: USB — wheel 파일 직접 복사 (rs-agent-env.tar.gz 불필요)

**① 인터넷 PC에서 — ML wheels 복사**

```bash
# 이 서버에 이미 있다면 바로 USB로 복사
cp -r RS-Agent/packages/ml-wheels/ /media/usb/ml-wheels/

# 없으면 먼저 다운로드
bash RS-Agent/scripts/download_ml.sh
cp -r RS-Agent/packages/ml-wheels/ /media/usb/ml-wheels/
```

**② USB 구조**

```
/usb/
├── RS-Agent/               ← 저장소 (git clone 또는 zip, ~300 MB)
│   └── packages/wheels/    ← 코어 패키지 (git에 포함됨)
└── ml-wheels/              ← ML 패키지 whl 파일 (~2.8 GB, 86개 파일)
    ├── torch-*.whl
    ├── transformers-*.whl
    └── ...
```

**③ 폐쇄망 PC에서 설치 (한 번에 완료)**

```bash
bash RS-Agent/scripts/setup_offline.sh /media/usb/ml-wheels
```

`rs-agent-env.tar.gz` 생성/전송 단계 없이 wheel → venv 직접 설치합니다.

---

**저장소 자체에 이미 포함된 항목 (별도 준비 불필요):**
- `install/Miniconda3-latest-Linux-x86_64.sh.partaa/ab` — Miniconda 설치 분할파일
- `packages/wheels/` — 코어 pip 패키지 wheel 파일 (~47개)
- `models/all-MiniLM-L6-v2/` — 임베딩 모델

설치 완료 후 자동 생성되는 실행 스크립트:

| 파일 | 설명 |
|------|------|
| `run_server.sh` | LLM 서버 (Mock 모드, GPU 불필요) |
| `run_server_gpu.sh` | LLM 서버 (A100 GPU 모드) |
| `run_ui.sh` | Gradio Web UI |

---

## 실행 방법

### 폐쇄망

```bash
# [터미널 1] LLM 서버 — Mock 모드 (GPU 불필요)
bash run_server.sh

# [터미널 1] A100 GPU 서버 — FP16 (~24 GB VRAM)
bash run_server_gpu.sh models/Qwen2.5-7B-Instruct

# [터미널 1] A100 GPU 서버 — 4-bit 양자화 (~12 GB VRAM)
bash run_server_gpu.sh models/Qwen2.5-7B-Instruct --load-in-4bit

# [터미널 2] Web UI
bash run_ui.sh
# 브라우저: http://localhost:7860
```

### 외부망

```bash
cp .env.example .env
# .env에 ANTHROPIC_API_KEY 입력
python app.py
```

### CLI

```bash
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

## 폐쇄망 구성 요소 체크리스트

| 구성 요소 | 위치 | git 포함 | 비고 |
|-----------|------|---------|------|
| Miniconda 설치 파일 | `install/Miniconda3-*.sh` | ✅ | 155 MB |
| 코어 pip wheels | `packages/wheels/` | ✅ | 104 MB, 파일별 <50 MB |
| 임베딩 모델 | `models/all-MiniLM-L6-v2/` | ✅ | 87 MB |
| venv 환경 전체 | `rs-agent-env.tar.gz` | ❌ | ~3 GB, SFTP 또는 USB로 전송 |
| 지식/솔루션 데이터 | `data/knowledge/`, `data/solutions/` | ✅ | JSON 파일 |
| 예시 이미지 | `data/images/` | ✅ | 위성 사진 |
| LLM 모델 (선택) | `models/<모델명>/` | ❌ | Mock 모드는 불필요 |

> `rs-agent-env.tar.gz`(torch + CUDA 포함 전체 환경)은 3 GB로 git 저장이 불가능합니다.
> torch wheel 단독으로도 500 MB이고, CUDA 라이브러리 포함 시 최대 파일 크기가 500 MB를 초과합니다.

---

## 구성 파일

`config.yaml`에서 다음을 설정합니다:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `llm.backend` | `openai_compatible` | `anthropic` / `openai_compatible` |
| `llm.base_url` | `http://localhost:11434/v1` | 로컬 서버 주소 |
| `llm.model` | `mock-rs-agent` | 모델 이름 |
| `knowledge_space.local_model_path` | `models/all-MiniLM-L6-v2` | 임베딩 모델 경로 |

---

## Novel Contributions

### Task-Aware Retrieval
Expert solution templates are retrieved based on semantic similarity to the user's query.

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
