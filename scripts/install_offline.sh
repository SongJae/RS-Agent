#!/usr/bin/env bash
# =============================================================================
# install_offline.sh  —  폐쇄망 PC에서 실행
#
# rs-agent-env.tar.gz를 압축 해제하고 RS-Agent를 바로 실행할 수 있게 설정합니다.
# 인터넷 연결이 전혀 없어도 됩니다.
#
# 전송 구조 (SFTP 또는 USB):
#   ~/
#   ├── rs-agent-env.tar.gz      ← venv 환경 전체 (~3 GB)
#   └── RS-Agent/                ← 저장소 (코드 + 모델 + wheels 포함)
#       ├── packages/wheels/     ← 코어 pip 패키지
#       ├── models/all-MiniLM-L6-v2/
#       └── scripts/install_offline.sh  ← 이 파일
#
# 사용법:
#   bash RS-Agent/scripts/install_offline.sh
#   bash RS-Agent/scripts/install_offline.sh /경로/rs-agent-env.tar.gz
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# tar.gz 위치: 인수로 지정하거나 RS-Agent/ 상위 디렉토리에서 자동 탐색
if [ $# -ge 1 ]; then
  ENV_TAR="$1"
else
  ENV_TAR="$(dirname "$PROJECT_DIR")/rs-agent-env.tar.gz"
  [ -f "$PROJECT_DIR/rs-agent-env.tar.gz" ] && ENV_TAR="$PROJECT_DIR/rs-agent-env.tar.gz"
fi

INSTALL_DIR="$HOME/rs-agent-env"

echo "======================================================"
echo " RS-Agent 오프라인 설치"
echo " venv 환경  : $ENV_TAR"
echo " 설치 경로  : $INSTALL_DIR"
echo " 프로젝트   : $PROJECT_DIR"
echo "======================================================"

# ── 1. tar.gz 확인 ───────────────────────────────────────────────────────────
if [ ! -f "$ENV_TAR" ]; then
  echo "[ERROR] venv 환경 파일을 찾을 수 없습니다: $ENV_TAR"
  echo ""
  echo "  확인 사항:"
  echo "    1) rs-agent-env.tar.gz를 홈 디렉토리(~/) 또는 RS-Agent/ 에 복사"
  echo "    2) bash install_offline.sh /경로/rs-agent-env.tar.gz"
  exit 1
fi

# ── 2. venv 환경 압축 해제 ───────────────────────────────────────────────────
echo ""
echo "[1/4] venv 환경 압축 해제 중 (시간 소요)..."
[ -d "$INSTALL_DIR" ] && rm -rf "$INSTALL_DIR"
# pack_conda_env.sh 는 $HOME 기준으로 rs-agent-env/ 를 아카이빙
# → $HOME 에 직접 풀어야 $HOME/rs-agent-env/ 가 생성됨
tar -xzf "$ENV_TAR" -C "$HOME"
echo "  → 압축 해제 완료: $INSTALL_DIR"

# ── 3. 경로 재설정 (포맷 자동 감지) ─────────────────────────────────────────
echo ""
echo "[2/4] 환경 경로 재설정 중..."

if [ -f "$INSTALL_DIR/bin/conda-unpack" ]; then
  # ── conda-pack 포맷: conda-unpack 이 경로를 자동으로 재설정 ──────────────
  echo "  conda-pack 포맷 감지 → conda-unpack 실행 중..."
  "$INSTALL_DIR/bin/conda-unpack"
  echo "  → 완료"
else
  # ── venv + tar 포맷: sed 로 경로 수동 치환 ───────────────────────────────
  ORIGINAL_PATH=$(cat "$INSTALL_DIR/.rs_original_path" 2>/dev/null || echo "")
  if [ -n "$ORIGINAL_PATH" ] && [ "$ORIGINAL_PATH" != "$INSTALL_DIR" ]; then
    echo "  venv 포맷 감지"
    echo "  원본: $ORIGINAL_PATH → 현재: $INSTALL_DIR"
    find "$INSTALL_DIR/bin" -maxdepth 1 -type f | while IFS= read -r f; do
      head -c 2 "$f" 2>/dev/null | grep -q $'#!' || continue
      sed -i "s|${ORIGINAL_PATH}|${INSTALL_DIR}|g" "$f" 2>/dev/null || true
    done
    for act in activate activate.csh activate.fish; do
      [ -f "$INSTALL_DIR/bin/$act" ] && \
        sed -i "s|${ORIGINAL_PATH}|${INSTALL_DIR}|g" "$INSTALL_DIR/bin/$act" 2>/dev/null || true
    done
    echo "  → 경로 재설정 완료"
  else
    echo "  → 경로 동일 — 재설정 불필요"
  fi
fi

# ── 4. 코어 패키지 보완 (wheels에서 설치) ────────────────────────────────────
WHEELS_DIR="$PROJECT_DIR/packages/wheels"
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]; then
  echo ""
  echo "[3/4] 코어 패키지 보완 설치 (로컬 wheels)..."
  "$INSTALL_DIR/bin/pip" install \
    numpy scikit-learn faiss-cpu Pillow PyYAML rank-bm25 rich python-dotenv requests \
    openai anthropic fastapi "uvicorn[standard]" tiktoken \
    --no-index --find-links "$WHEELS_DIR" -q 2>/dev/null || true
  echo "  → 완료"
else
  echo "[3/4] wheels 디렉토리 없음 — 건너뜀"
fi

# ── 5. 실행 스크립트 생성 ─────────────────────────────────────────────────────
echo ""
echo "[4/4] 실행 스크립트 생성 중..."

cat > "$PROJECT_DIR/run_server.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — Mock 모드 (GPU 불필요)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$INSTALL_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend mock --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_server_gpu.sh" <<RUNEOF
#!/usr/bin/env bash
# LLM 서버 — GPU 모드 (실제 LLM 추론)
# 사용법: bash run_server_gpu.sh /models/Qwen2.5-7B-Instruct [--load-in-4bit]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$INSTALL_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python scripts/start_local_server.py --backend transformers \
  --model-path "\${1:?모델 경로가 필요합니다}" \${2:-} --port 11434
RUNEOF

cat > "$PROJECT_DIR/run_ui.sh" <<RUNEOF
#!/usr/bin/env bash
# Gradio Web UI
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$INSTALL_DIR/bin:\$PATH"
cd "$PROJECT_DIR"
python app.py --port 7860
RUNEOF

chmod +x "$PROJECT_DIR/run_server.sh" \
         "$PROJECT_DIR/run_server_gpu.sh" \
         "$PROJECT_DIR/run_ui.sh"

echo "  → run_server.sh      생성 완료"
echo "  → run_server_gpu.sh  생성 완료"
echo "  → run_ui.sh          생성 완료"

# ── 완료 ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 설치 완료!"
echo ""
echo " [터미널 1] LLM 서버 시작 (Mock 모드)"
echo "   bash $PROJECT_DIR/run_server.sh"
echo ""
echo " [터미널 1] A100 GPU 서버 (FP16, ~24 GB VRAM)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct"
echo ""
echo " [터미널 1] A100 GPU 서버 (4-bit, ~12 GB VRAM)"
echo "   bash $PROJECT_DIR/run_server_gpu.sh models/Qwen2.5-7B-Instruct --load-in-4bit"
echo ""
echo " [터미널 2] Web UI 시작"
echo "   bash $PROJECT_DIR/run_ui.sh"
echo ""
echo " 브라우저: http://localhost:7860"
echo "======================================================"
