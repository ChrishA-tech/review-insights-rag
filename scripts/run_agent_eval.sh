#!/usr/bin/env bash
# Run the LLM agent evaluation locally with Ollama (https://ollama.com). Free, no API key.
#   bash scripts/run_agent_eval.sh            # default model qwen2.5:7b
#   MODEL=llama3.1:8b bash scripts/run_agent_eval.sh
set -e
MODEL=${MODEL:-qwen2.5:7b}
PY=${PY:-python3}
command -v ollama >/dev/null || { echo "Install Ollama first: brew install ollama  (or https://ollama.com/download)"; exit 1; }
curl -fsS localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 3
ollama pull "$MODEL"
# isolated environment, so nothing is installed into the system Python
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[dev]"
# projects whose tools read raw data need it locally
if grep -q "reviews.csv" scripts/get_data.sh 2>/dev/null; then bash scripts/get_data.sh; fi
python eval/run_eval.py --backend ollama --model "$MODEL"
python eval/run_eval.py --backend ollama --model "$MODEL" --no-verifier
echo "Results in reports/agent_eval_*.json. Paste the two summaries back to update the README and CV."
