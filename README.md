# PoemBench

PoemBench 是一个中文古典诗词基准测试框架，用于评估大语言模型在诗词理解任务上的表现。

**[Contributors 林雨夜]**<sup>1*†</sup>&ensp; http://lyy0323.space
**[Contributors 抱木]**<sup>2</sup>&ensp;

**致谢：** 上海交大国学社全体驻站诗人

<sup>1</sup>SJTU&emsp;<sup>2</sup>IIE

<br>
<small>* <b>Dataset Collection Lead</b> &emsp; † <b>Q&A Design Lead</b></small>

[![Paper]()](link)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](link)

</div>


## 特性

- 支持本地模型（HuggingFace Transformers）和多种 API 模型
- 提供指令遵循率（IFR）和指令遵循下准确率（Acc|IF）两个关键指标
- 支持随机抽样评估，保证可重复性（默认种子 1127）
- 结果按模型名称分目录保存

## 安装

```bash
pip install -r requirements.txt

# 根据需要安装 API 客户端
pip install openai          # OpenAI / 兼容 API
pip install anthropic       # Anthropic Claude
pip install zhipuai         # 智谱 GLM
pip install dashscope       # 阿里云 Qwen
```

## 快速开始

### 评估本地模型

```bash
# 基础用法
python scripts/eval_local.py --model-path ./models/Qwen3-4B

# 使用配置文件
python scripts/eval_local.py --config configs/models/qwen3_4b_local.yaml

# 随机抽样 100 个样本
python scripts/eval_local.py --model-path ./models/Qwen3-4B --sample-n 100
```

### 评估 API 模型

```bash
# 使用配置文件
python scripts/eval_api.py --config configs/models/openai_gpt4o.yaml

# 命令行指定参数
python scripts/eval_api.py --model-type openai --model-name gpt-4o

# 随机抽样
python scripts/eval_api.py --config configs/models/openai_gpt4o.yaml --sample-n 100
```

### 批量评估

```bash
# 评估目录下所有模型配置
python scripts/eval_batch.py --config-dir configs/models/

# 指定多个配置文件
python scripts/eval_batch.py --configs configs/models/qwen3_4b_local.yaml configs/models/openai_gpt4o.yaml

# 带抽样的批量评估
python scripts/eval_batch.py --config-dir configs/models/ --sample-n 100 --seed 1127
```

### 查看数据集

```bash
python scripts/list_datasets.py
python scripts/list_datasets.py --corpus tang --task guess_author
```

## 添加新模型

### 添加本地模型

在 `configs/models/` 目录下创建 YAML 配置文件：

```yaml
# configs/models/my_local_model.yaml
model_name: "my-model"
model_type: "local"
model_path: "./models/MyModel"  # HuggingFace 模型路径
device: "auto"                   # auto, cuda, cpu
torch_dtype: "float16"           # float16, bfloat16, float32

# 生成参数
max_new_tokens: 128
temperature: 0.1
top_p: 0.9
```

### 添加 API 模型

#### OpenAI / 兼容 API

```yaml
# configs/models/my_openai_model.yaml
model_name: "gpt-4o"
model_type: "openai"
api_key: "$OPENAI_API_KEY"      # 使用环境变量
api_model_name: "gpt-4o"
# api_base: "https://custom-endpoint.com/v1"  # 可选：自定义端点

max_new_tokens: 128
temperature: 0.1
```

#### Anthropic Claude

```yaml
# configs/models/claude.yaml
model_name: "claude-3-5-sonnet"
model_type: "anthropic"
api_key: "$ANTHROPIC_API_KEY"
api_model_name: "claude-3-5-sonnet-20241022"

max_new_tokens: 128
temperature: 0.1
```

#### 智谱 GLM

```yaml
# configs/models/glm.yaml
model_name: "glm-4"
model_type: "zhipu"
api_key: "$ZHIPUAI_API_KEY"
api_model_name: "glm-4"

max_new_tokens: 128
temperature: 0.1
```

#### 阿里云 Qwen

```yaml
# configs/models/qwen_api.yaml
model_name: "qwen-turbo"
model_type: "qwen_api"
api_key: "$DASHSCOPE_API_KEY"
api_model_name: "qwen-turbo"

max_new_tokens: 128
temperature: 0.1
```

#### DeepSeek

```yaml
# configs/models/deepseek.yaml
model_name: "deepseek-chat"
model_type: "deepseek"
api_key: "$DEEPSEEK_API_KEY"
api_base: "https://api.deepseek.com"
api_model_name: "deepseek-chat"

max_new_tokens: 128
temperature: 0.1
```

### 添加自定义 API 模型

1. 在 `src/models/api_model.py` 中继承 `APIModel` 类：

```python
class MyCustomModel(APIModel):
    def load(self) -> None:
        # 初始化 API 客户端
        self.client = MyClient(api_key=self.config.api_key)
        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        start_time = time.time()
        try:
            response = self.client.generate(prompt)
            return GenerationResult(
                response=response.text,
                latency=time.time() - start_time,
                success=True
            )
        except Exception as e:
            return GenerationResult(
                response="",
                latency=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
```

2. 在 `src/models/registry.py` 中注册：

```python
class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {
        # ... 现有模型
        "my_custom": MyCustomModel,
    }
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **IFR** (Instruction Following Rate) | 指令遵循率：模型回复符合预期格式的比例 |
| **Acc\|IF** (Accuracy if Followed) | 指令遵循下准确率：在格式正确的回复中，答案正确的比例 |
| **Accuracy** | 总体准确率：正确答案数 / 总样本数 |

输出示例：
```
Dataset                             IFR        Acc|IF     Accuracy   Correct/Total
----------------------------------------------------------------------------------------------------
tang.guess_author.standard          98.50%     45.20%     44.52%     223/500
tang.guess_word.standard            99.20%     52.10%     51.68%     258/500
----------------------------------------------------------------------------------------------------
Overall                             98.85%     48.65%     48.10%     481/1000
```

## 数据集

### 命名规则

`{corpus}.{task}.{variant}.jsonl`

- **Corpus**: tang, song, tang300, tangsong, today
- **Task**: guess_author, guess_word, guess_ci_tone, match_sentence, sort_poem
- **Variant**: standard, fewshot1/3/10, cot, couplets, jue, lyu

### 任务类型

| 任务 | 类型 | 说明 |
|------|------|------|
| guess_author | 选择题 | 猜测诗句作者 |
| guess_word | 选择题 | 填空选词 |
| guess_ci_tone | 选择题 | 猜测词牌名 |
| match_sentence | 选择题 | 匹配上下句 |
| sort_poem | 排序题 | 诗句排序（绝句4句/律诗8句）|

## 结果输出

评估结果保存在 `results/{model_name}/` 目录：

```
results/
└── gpt-4o/
    ├── eval_20241128_143052.json  # 完整评估结果（含详细数据）
    └── summary.txt                 # 可读的汇总表格
```

## 项目结构

```
poembanch/
├── src/
│   ├── models/              # 模型实现
│   │   ├── base.py          # 基类 BaseModel, ModelConfig
│   │   ├── local_model.py   # 本地模型（HuggingFace）
│   │   ├── api_model.py     # API 模型
│   │   └── registry.py      # 模型注册表
│   ├── evaluation/          # 评估流水线
│   │   ├── dataset.py       # 数据集加载（支持抽样）
│   │   ├── prompt.py        # Prompt 构建与响应解析
│   │   ├── metrics.py       # 指标计算（IFR, Acc|IF）
│   │   └── pipeline.py      # 评估流水线
│   └── utils/
│       ├── config.py        # 配置加载
│       └── logger.py        # 日志工具
├── scripts/                 # CLI 脚本
│   ├── eval_local.py        # 评估本地模型
│   ├── eval_api.py          # 评估 API 模型
│   ├── eval_batch.py        # 批量评估
│   └── list_datasets.py     # 列出数据集
├── configs/
│   ├── models/              # 模型配置
│   └── eval/                # 评估配置
├── data/                    # 数据集
└── results/                 # 评估结果
```
