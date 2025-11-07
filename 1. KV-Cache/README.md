# Custom KV-Cache Implementation

A custom implementation of KV-Cache optimization for efficient LLM inference, inspired by [this tutorial](https://apetulante.github.io/posts/KV-Caching/kv_caching.html).

## 📖 Explanation

**First, read the detailed instructions here:**
- 🇬🇧 **[English Version](INSTRUCTIONS.md#english)** - How KV Cache Works
- 🇻🇳 **[Tiếng Việt](INSTRUCTIONS.md#tiếng-việt)** - Cách KV Cache Hoạt Động

## Overview

KV-Cache is an optimization technique that caches key-value pairs from previous tokens during autoregressive generation. This reduces computation from O(n²) to O(n) per token, making inference significantly faster.

![KV Cache Visualization](images/KV_Cache.jpg)

This implementation provides **full control** over cache management, allowing you to modify, inspect, or optimize the cache as needed.

## Module Structure

- **`kv_cache.py`**: Core KV-Cache implementation
  - `KVCache`: Single-layer cache class
  - `MultiLayerKVCache`: Multi-layer wrapper for HuggingFace models

- **`generation.py`**: Text generation functions
  - `generate_without_cache()`: Baseline generation without caching
  - `generate_with_kv_cache()`: Efficient generation with custom KV-Cache

- **`benchmark.py`**: Performance benchmarking utilities
  - Compare generation speed with/without Custom KV-Cache

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simple Usage (Works with HuggingFace Model, tested with Qwen in GQA)

```python
import os
from generation import generate_with_kv_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify model directory (models will be downloaded here)
model_dir = "./models"  # Change to your preferred directory
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# Create directory if needed
os.makedirs(model_dir, exist_ok=True)

# Load any model (Qwen, LLaMA, GPT, etc.)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=model_dir  # Models will be saved here
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_dir
)

# Generate with Custom KV-Cache (works with HuggingFace model)
text, time_taken = generate_with_kv_cache(
    model, tokenizer, 
    prompt="The future of AI is",
    max_length=50,
    temperature=0.7,
    max_seq_len=2048  # Custom cache size
)
print(f"Generated: {text}")
print(f"Time: {time_taken:.2f}s")
```

### Compare with/without Cache

```python
from generation import generate_without_cache, generate_with_kv_cache

# Without cache (baseline)
text1, time1 = generate_without_cache(model, tokenizer, prompt, max_length=50)

# With Custom KV-Cache (optimized)
text2, time2 = generate_with_kv_cache(model, tokenizer, prompt, max_length=50)

print(f"Speedup: {time1/time2:.2f}x")
```

### Direct Cache Access

```python
from kv_cache import MultiLayerKVCache
from generation import generate_with_kv_cache

# The cache is managed internally, but you can access it for custom operations
# by modifying the generation function or creating your own wrapper
```


## Running Examples

```bash
# Simple example (works with HuggingFace model)
python example_simple.py

# Benchmark performance
python benchmark.py
```

## Key Benefits

1. **Speed**: 2-5x faster generation for long sequences
2. **Efficiency**: O(n) computation per token instead of O(n²)
3. **Memory**: Reuses computed K,V pairs instead of recomputing
4. **Full Control**: Modify, inspect, or optimize cache as needed
5. **Scalability**: Essential for real-time applications

## Custom Cache Features

- **Multi-layer support**: Handles all transformer layers automatically
- **Format conversion**: Seamlessly converts between custom cache and HuggingFace format
- **Flexible**: Easy to extend with compression, pruning, or quantization
- **Debugging**: Inspect cache contents for analysis

## References

- [KV Caching Tutorial](https://apetulante.github.io/posts/KV-Caching/kv_caching.html)

