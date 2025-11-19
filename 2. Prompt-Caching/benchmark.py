"""
Benchmark and Performance Comparison for Prompt Caching

This module provides benchmarking utilities to compare generation
performance with and without Prompt Caching, demonstrating the speedup benefits
for shared prefixes.
"""

import os
import torch
import time
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_cache import PromptCacheManager
from generation import generate_with_prompt_cache

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Install with 'pip install matplotlib' to enable plotting.")


def benchmark_scaling(
    model_name: str,
    base_system_prompt: str,
    query: str,
    prompt_lengths: List[int],
    max_new_tokens: int = 50,
    device: str = "auto",
    model_dir: str = "./models",
    max_seq_len: int = 4096
) -> None:
    """
    Benchmark Prompt Caching performance across different prompt lengths.
    """
    print("="*70)
    print("Prompt Caching Scaling Benchmark")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Prompt Lengths: {prompt_lengths}")
    print("="*70)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir=model_dir
    )
    model.eval()
    print(f"Model loaded on {device}\n")
    
    cache_manager = PromptCacheManager(device=device)
    results = []
    
    # We need a long source text to generate prompts of different lengths
    # Let's just repeat the base prompt enough times
    long_source = base_system_prompt * 100
    
    for length in prompt_lengths:
        print(f"\n{'='*70}")
        print(f"Benchmarking with System Prompt Length ~{length} chars")
        print(f"{'='*70}")
        
        current_system_prompt = long_source[:length]
        current_system_prompt = long_source[:length]
        
        # 1. Baseline (No Cache)
        full_prompt_test = f"{current_system_prompt}\nUser: {query}\nAssistant:"
        
        print(f"  Without Cache...", end=" ", flush=True)
        cache_manager.clear()
        try:
            _, time_no_cache = generate_with_prompt_cache(
                model, tokenizer, full_prompt_test, cache_manager, max_length=max_new_tokens, device=device, max_seq_len=max_seq_len
            )
            print(f"[OK] {time_no_cache:.2f}s")
        except Exception as e:
            print(f"[Error] {e}")
            time_no_cache = None

        # 2. With Cache
        # Warmup with a DIFFERENT query to ensure we only cache the System Prompt part
        # and not the specific user query we are about to test.
        full_prompt_warmup = f"{current_system_prompt}\nUser: Warmup Query\nAssistant:"
        
        cache_manager.clear()
        generate_with_prompt_cache(
            model, tokenizer, full_prompt_warmup, cache_manager, max_length=1, device=device, max_seq_len=max_seq_len
        )
        
        print(f"  With Cache...", end=" ", flush=True)
        try:
            # Now run with the actual test query. 
            # It should match the System Prompt prefix from the warmup, but not the User Query part.
            _, time_with_cache = generate_with_prompt_cache(
                model, tokenizer, full_prompt_test, cache_manager, max_length=max_new_tokens, device=device, max_seq_len=max_seq_len
            )
            print(f"[OK] {time_with_cache:.2f}s")
        except Exception as e:
            print(f"[Error] {e}")
            time_with_cache = None
            
        if time_no_cache and time_with_cache:
            speedup = time_no_cache / time_with_cache
            results.append((length, time_no_cache, time_with_cache, speedup))
        else:
            results.append((length, time_no_cache, time_with_cache, None))

    # Print Summary
    print("\n" + "="*70)
    print("Scaling Performance Summary")
    print("="*70)
    print(f"{'Length':<10} {'No Cache (s)':<15} {'With Cache (s)':<15} {'Speedup':<10}")
    print("-"*70)
    for length, t_no, t_yes, speedup in results:
        if speedup:
            print(f"{length:<10} {t_no:<15.2f} {t_yes:<15.2f} {speedup:<10.2f}x")
        else:
            print(f"{length:<10} {'Error':<15} {'Error':<15} {'-':<10}")
            
    if HAS_MATPLOTLIB and len(results) > 0:
        plot_scaling_results(results, model_name)

def plot_scaling_results(results: List[tuple], model_name: str):
    """Plot scaling benchmark results."""
    if not HAS_MATPLOTLIB:
        return
        
    lengths = [r[0] for r in results if r[3]]
    time_no = [r[1] for r in results if r[3]]
    time_yes = [r[2] for r in results if r[3]]
    speedups = [r[3] for r in results if r[3]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time vs Length
    ax1.plot(lengths, time_no, 'o-', label='No Cache', color='red')
    ax1.plot(lengths, time_yes, 's-', label='With Prompt Cache', color='green')
    ax1.set_xlabel('System Prompt Length (chars)')
    ax1.set_ylabel('Generation Time (s)')
    ax1.set_title(f'Generation Time vs Prompt Length\n{model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Length
    ax2.plot(lengths, speedups, 'o-', label='Speedup', color='blue')
    ax2.axhline(y=1.0, color='gray', linestyle='--')
    ax2.set_xlabel('System Prompt Length (chars)')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs Prompt Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for l, s in zip(lengths, speedups):
        ax2.annotate(f'{s:.2f}x', (l, s), xytext=(0, 5), textcoords='offset points', ha='center')
        
    plt.tight_layout()
    plt.show()

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Base text to repeat
    base_system_prompt = (
        "You are a helpful AI assistant. " + 
        "The history of computing is longer than the history of computing hardware and modern computing technology and includes the history of methods intended for pen and paper or for chalk and slate, with or without the aid of tables. "
    )
    
    query = "What is the future of AI?"
    
    # Benchmark with increasing prompt lengths
    # Note: These are char lengths, roughly /4 for tokens
    lengths = [500, 1000, 2000, 4000, 8000] 
    
    benchmark_scaling(model_name, base_system_prompt, query, lengths, max_seq_len=4096)

if __name__ == "__main__":
    main()
