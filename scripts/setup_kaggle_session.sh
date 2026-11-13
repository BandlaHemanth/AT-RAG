#!/usr/bin/env bash
# Restore the TAE-R working session on a fresh Kaggle GPU notebook (idempotent).
#
#   1. main Python: umap-learn / hdbscan versions used to fit the topic models
#   2. Ollama server with the pinned LLMs; "gpt-4o" alias -> qwen2.5:7b-instruct-q8_0
#   3. Python 3.12.3 environment for the unedited original AT-RAG code
#      (scientific pins from the backed-up training freeze, torch 2.14.1 CUDA 13 build,
#      LangChain 0.3 integration versions)
#
# Usage: bash scripts/setup_kaggle_session.sh /path/to/at-rag-baseline-requirements.txt
set -euo pipefail
FREEZE="${1:?path to at-rag-baseline-requirements.txt}"
V=/kaggle/working/atrag312
LOG=/kaggle/working/setup_session.log
step() { echo "== $*"; }

step "main environment: umap-learn 0.5.12, hdbscan 0.8.44"
python3 -m pip install -q umap-learn==0.5.12 hdbscan==0.8.44 >>"$LOG" 2>&1

step "ollama"
if ! command -v ollama >/dev/null; then
  (apt-get -qq update && apt-get -qq install -y zstd) >>"$LOG" 2>&1 || true
  curl -fsSL https://ollama.com/install.sh | sh >>"$LOG" 2>&1
fi
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1 \
    setsid nohup ollama serve >/kaggle/working/ollama.log 2>&1 < /dev/null &
  for _ in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi
ollama pull qwen2.5:7b-instruct-q8_0 >>"$LOG" 2>&1
ollama pull qwen2.5:14b-instruct-q4_K_M >>"$LOG" 2>&1
ollama rm gpt-4o >>"$LOG" 2>&1 || true
ollama cp qwen2.5:7b-instruct-q8_0 gpt-4o >>"$LOG" 2>&1

step "python 3.12.3 environment for the original code"
if [ ! -x "$V/bin/python" ]; then
  python3 -m pip install -q uv >>"$LOG" 2>&1
  uv python install 3.12.3 >>"$LOG" 2>&1
  uv venv -q --python 3.12.3 "$V" >>"$LOG" 2>&1
  grep -iE '^(numpy|numba|llvmlite|scipy|scikit-learn|umap-learn|hdbscan|bertopic|transformers|sentence-transformers|cloudpickle|joblib|tokenizers|huggingface-hub|pandas)==' "$FREEZE" \
    > /kaggle/working/sci_constraints.txt
  echo "torch==2.14.1" >> /kaggle/working/sci_constraints.txt
  uv pip install -q --python "$V/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 >>"$LOG" 2>&1
  uv pip install -q --python "$V/bin/python" -c /kaggle/working/sci_constraints.txt \
    $(grep -v '^torch' /kaggle/working/sci_constraints.txt | tr '\n' ' ') >>"$LOG" 2>&1
  uv pip install -q --python "$V/bin/python" -c /kaggle/working/sci_constraints.txt \
    langchain==0.3.2 langchain-community==0.3.1 chromadb==0.5.11 python-decouple==3.8 \
    openai==1.51.2 httpx==0.27.2 posthog==3.7.0 langgraph==0.2.34 >>"$LOG" 2>&1
fi

step "versions"
python3 -c "import umap, hdbscan, sklearn, numpy; print('main:', umap.__version__, sklearn.__version__, numpy.__version__)"
"$V/bin/python" -c "import sys, torch, numpy, numba, bertopic, langchain; print('py312:', sys.version.split()[0], torch.__version__, 'cuda', torch.cuda.is_available(), numpy.__version__, numba.__version__, bertopic.__version__, langchain.__version__)"
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys; print('ollama:', {m['name']: m['digest'][:16] for m in json.load(sys.stdin)['models']})"
echo "setup complete"
