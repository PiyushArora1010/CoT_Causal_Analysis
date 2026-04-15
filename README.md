# CoT Causal Analysis

**causal analysis of chain-of-thought** by editing individual reasoning steps with a counterfactual and checking whether the final answer changes.

## File Structure
- `modules/model.py`: Hugging Face `transformers` wrapper (loads models with `trust_remote_code=True`).
- `modules/causal_engine.py`: generate CoT → edit a chosen step → regenerate answer; includes batched causality scoring.
- `modules/dataset.py`: GSM8K difficulty subset loader (`lime-nlp/GSM8K_Difficulty`) + a simple JSON dataset helper for BBQ dataset.
- `app.py`: Gradio UI for interactive CoT editing.
- `main.py`: batch run on GSM8K and write JSONL results.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets gradio tqdm numpy
```

## Run
Interactive UI:
```bash
python app.py
```
Batch evaluation (writes a `*.jsonl` file in the repo root):
```bash
python main.py
```

## Notes
- Defaults to `Qwen/Qwen3-4B`; change `model_tag` in `app.py`/`main.py` to use a different model.
- The engine expects model outputs containing `<think></think>` tags.
