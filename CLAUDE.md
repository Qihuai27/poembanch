# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PoemBench is a Chinese poetry benchmark evaluation framework for testing LLM performance on classical Chinese poetry comprehension tasks. It evaluates models on five task types using Tang and Song dynasty poetry.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt

# For API models, install additional dependencies:
pip install openai          # OpenAI/compatible APIs
pip install anthropic       # Anthropic Claude
pip install zhipuai         # Zhipu GLM
pip install dashscope       # Alibaba Qwen
```

**Evaluate a local model:**
```bash
python scripts/eval_local.py --model-path ./models/Qwen3-4B
python scripts/eval_local.py --config configs/models/qwen3_4b_local.yaml
python scripts/eval_local.py --model-path ./models/Qwen3-4B --sample-n 100  # Random sample 100
```

**Evaluate an API model:**
```bash
python scripts/eval_api.py --config configs/models/openai_gpt4o.yaml
python scripts/eval_api.py --model-type openai --model-name gpt-4o
python scripts/eval_api.py --config configs/models/openai_gpt4o.yaml --sample-n 100
```

**Batch evaluate multiple models:**
```bash
python scripts/eval_batch.py --config-dir configs/models/
python scripts/eval_batch.py --configs configs/models/qwen3_4b_local.yaml configs/models/openai_gpt4o.yaml
python scripts/eval_batch.py --config-dir configs/models/ --sample-n 100 --seed 1127
```

**List available datasets:**
```bash
python scripts/list_datasets.py
python scripts/list_datasets.py --corpus tang --task guess_author
```

## Metrics

The evaluation reports three key metrics:

- **IFR (Instruction Following Rate)**: Percentage of responses that follow the expected format (e.g., returning a number 1-4 for multiple choice)
- **Acc|IF (Accuracy if Followed)**: Accuracy among responses that followed instructions correctly
- **Accuracy**: Overall accuracy (correct / total samples)

## Sampling

Use `--sample-n N` to randomly sample N samples per dataset with fixed seed (default: 1127):
```bash
python scripts/eval_local.py --model-path ./models/Qwen3-4B --sample-n 100 --seed 1127
```

## Architecture

```
poembanch/
├── src/
│   ├── models/           # Model implementations
│   │   ├── base.py       # BaseModel abstract class, ModelConfig
│   │   ├── local_model.py    # HuggingFace local model
│   │   ├── api_model.py      # API models (OpenAI, Anthropic, Zhipu, Qwen, DeepSeek)
│   │   └── registry.py       # Model registry and factory
│   ├── evaluation/       # Evaluation pipeline
│   │   ├── dataset.py    # DatasetLoader, TaskSample (with sampling support)
│   │   ├── prompt.py     # PromptBuilder, ResponseParser (with format validation)
│   │   ├── metrics.py    # MetricsCalculator, EvaluationMetrics (IFR, Acc|IF)
│   │   └── pipeline.py   # EvaluationPipeline
│   └── utils/
│       ├── config.py     # Config loading/saving
│       └── logger.py     # Logging setup
├── scripts/              # CLI scripts
│   ├── eval_local.py     # Evaluate local model
│   ├── eval_api.py       # Evaluate API model
│   ├── eval_batch.py     # Batch evaluation
│   └── list_datasets.py  # List datasets
├── configs/
│   ├── models/           # Model configurations
│   └── eval/             # Evaluation configurations
├── data/                 # Benchmark datasets
└── results/{model_name}/ # Evaluation outputs per model
```

### Task Types
- **multiple_choice**: guess_author, guess_word, match_sentence, guess_ci_tone
- **sorting**: sort_poem (绝句 4 lines, 律诗 8 lines)

### Dataset Naming Convention
`{corpus}.{task}.{variant}.jsonl`
- Corpus: tang, song, tang300, tangsong, today
- Task: guess_author, guess_word, guess_ci_tone, match_sentence, sort_poem
- Variant: standard, fewshot1/3/10, cot, couplets, jue, lyu

### Data Format (JSONL)
```json
{
  "task_id": "tang.guess_author.standard.0",
  "type": "multiple_choice",
  "prompt": "谁写下了这句诗？",
  "demo": "人间荣耀因缘浅...",
  "choices": ["李白", "孟浩然", "白居易", "王昌龄"],
  "goal": "白居易",
  "hint": ""
}
```

### Result Format
Results are saved in `results/{model_name}/`:
- `eval_YYYYMMDD_HHMMSS.json`: Full evaluation results with detailed per-sample data
- `summary.txt`: Human-readable summary table

### Adding New Models

1. **Local Model**: Create config in `configs/models/`:
```yaml
model_name: "my-model"
model_type: "local"
model_path: "./models/MyModel"
device: "auto"
```

2. **API Model**: Create config with API settings:
```yaml
model_name: "my-api-model"
model_type: "openai"  # or anthropic, zhipu, qwen_api, deepseek
api_key: "$MY_API_KEY"  # Use environment variable
api_model_name: "model-name"
```

3. **Custom API**: Extend `APIModel` in `src/models/api_model.py` and register in `registry.py`.
