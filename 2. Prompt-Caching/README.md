# Prompt Caching Implementation

A custom implementation of Prompt Caching optimization for efficient LLM inference, enabling significant latency reduction for requests with shared prefixes (e.g., system prompts).

## 📖 Explanation

**First, read the detailed instructions here:**
- 🇬🇧 **[English Version](INSTRUCTIONS.md#english)** - How Prompt Caching Works
- 🇻🇳 **[Tiếng Việt](INSTRUCTIONS.md#tiếng-việt)** - Cách Prompt Caching Hoạt Động

## Overview

Prompt Caching is an optimization technique that stores the Key-Value (KV) states of frequently used prompt prefixes (like system instructions or document contexts). When a new request starts with a cached prefix, the model can skip the computation for that part, reducing Time To First Token (TTFT).

## Module Structure

- **`prompt_cache.py`**: Manages the storage and retrieval of KV caches.
  - `PromptCacheManager`: Handles cache storage, retrieval, and prefix matching.
- **`kv_cache.py`**: Core KV-Cache implementation (extended with truncation support).
- **`generation.py`**: Text generation functions.
  - `generate_with_prompt_cache()`: Generation logic that leverages cached prefixes.
- **`benchmark.py`**: Performance benchmarking utilities.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simple Usage

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_cache import PromptCacheManager
from generation import generate_with_prompt_cache

# Setup
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_dir = "./models"
cache_manager = PromptCacheManager()

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

# Define a shared system prompt
system_prompt = "You are a helpful AI assistant..."

# 1. First run (Cache Miss - computes and stores cache)
prompt1 = f"{system_prompt}\nUser: Hello!"
response1, time1 = generate_with_prompt_cache(model, tokenizer, prompt1, cache_manager)

# 2. Second run (Cache Hit - reuses system prompt cache)
prompt2 = f"{system_prompt}\nUser: How are you?"
response2, time2 = generate_with_prompt_cache(model, tokenizer, prompt2, cache_manager)

print(f"Speedup: {time1/time2:.2f}x")
```

## Running Benchmarks

```bash
python benchmark.py
```

## Key Benefits

1.  **Reduced Latency**: Significantly lower Time To First Token (TTFT) for cached prompts.
2.  **Efficiency**: Avoids redundant computation for static system prompts or documents.
3.  **Cost Savings**: Reduces compute resources required per request.
