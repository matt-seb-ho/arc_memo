# Setup Guide

Quick setup for ArcMemo experiments.

---

## 1. Virtual Environment

**Requirements:** Python 3.10+ (llmplus uses dataclass slots)

```bash
cd arc_memo

# Check Python version
python3 --version  # Must be 3.10+

# Create venv (first time only)
python3 -m venv .venv  # Or python3.11, python3.12

# Activate (every session)
source .venv/bin/activate

# Deactivate when done
deactivate
```

---

## 2. Install Dependencies

```bash
# Activate venv first
source .venv/bin/activate

# Core dependencies
pip install -r requirements.txt

# For AIME experiments
pip install datasets pandas

# For memory analysis
cd ../rebuttal/analysis
pip install sentence-transformers scikit-learn
cd ../../arc_memo
```

---

## 3. API Keys

```bash
# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF

# Verify
cat .env
```

---

## 4. Download AIME Dataset (if needed)

```bash
source .venv/bin/activate
cd data/aime
python download_and_prepare.py
cd ../..
```

---

## Quick Start

```bash
# Each session:
cd arc_memo
source .venv/bin/activate

# Then run experiments
bash experiments/run.sh
```

---

## Verify Setup

```bash
source .venv/bin/activate

# Check imports
python -c "from concept_mem.data.aime_math import load_aime_data; print('✓ AIME loader')"
python -c "from sentence_transformers import SentenceTransformer; print('✓ Memory analysis')"
python -c "import llmplus; print('✓ LLM client')"
```

