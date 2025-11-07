"""
Benchmark and Performance Comparison

This module provides benchmarking utilities to compare generation
performance with and without Custom KV-Cache, demonstrating the speedup benefits.

Reference: https://apetulante.github.io/posts/KV-Caching/kv_caching.html
"""

import os
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation import generate_without_cache, generate_with_kv_cache

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Install with 'pip install matplotlib' to enable plotting.")


def benchmark_generation(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "auto",
    model_dir: str = "./models"
) -> None:
    """
    Benchmark text generation with and without KV-Cache.
    
    This function demonstrates the performance benefits of KV-Cache by
    comparing generation times for the same prompt.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
        model_dir: Directory to download/store models (default: ./models)
    """
    print("="*60)
    print("KV-Cache Performance Benchmark")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Max tokens: {max_new_tokens}")
    print("="*60)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load model and tokenizer (will download to model_dir if not already present)
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        cache_dir=model_dir
    )
    model.eval()
    print(f"Model loaded on {device}\n")
    
    # Benchmark without cache
    print("1. Generating WITHOUT KV-Cache...")
    print("   (This recomputes everything for each token)")
    result_no_cache, time_no_cache = generate_without_cache(
        model, tokenizer, prompt, max_length=max_new_tokens, device=device, temperature=1.0
    )
    print(f"   Time: {time_no_cache:.2f}s")
    print(f"   Generated: {result_no_cache[:80]}...\n")
    
    # Benchmark with custom cache
    print("2. Generating WITH Custom KV-Cache...")
    print("   (This reuses cached K,V from previous tokens)")
    result_with_cache, time_with_cache = generate_with_kv_cache(
        model, tokenizer, prompt, max_length=max_new_tokens, device=device
    )
    print(f"   Time: {time_with_cache:.2f}s")
    print(f"   Generated: {result_with_cache[:80]}...\n")
    
    # Compare results
    print("="*60)
    print("Performance Comparison")
    print("="*60)
    if time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        time_saved = time_no_cache - time_with_cache
        print(f"Speedup: {speedup:.2f}x faster with Custom KV-Cache")
        print(f"Time saved: {time_saved:.2f}s")
        print(f"Efficiency gain: {(1 - time_with_cache/time_no_cache)*100:.1f}%")
    print("="*60)
    
    # Key insights
    print("\nKey Insights:")
    print("- Without caching: O(n²) computation per token")
    print("- With Custom KV-Cache: O(n) computation per token")
    print("- The longer the prompt, the more benefit from caching")
    print("- Custom KV-Cache provides full control over cache management")


def benchmark_scaling(
    model_name: str,
    prompt: str,
    token_lengths: Optional[List[int]] = None,
    device: str = "auto",
    model_dir: str = "./models",
    temperature: float = 1.0
) -> None:
    """
    Benchmark generation performance across different sequence lengths.
    
    This demonstrates how KV-Cache speedup scales with sequence length,
    showing that longer sequences benefit more from caching.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        token_lengths: List of token lengths to benchmark (default: [10, 20, 50, 100, 200])
        device: Device to run on
        model_dir: Directory to download/store models
        temperature: Sampling temperature
    """
    if token_lengths is None:
        token_lengths = [10, 20, 50, 100, 200]
    
    print("="*70)
    print("KV-Cache Scaling Benchmark")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Token lengths: {token_lengths}")
    print("="*70)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        cache_dir=model_dir
    )
    model.eval()
    print(f"Model loaded on {device}\n")
    
    results = []
    
    for max_tokens in token_lengths:
        print(f"\n{'='*70}")
        print(f"Benchmarking with {max_tokens} tokens")
        print(f"{'='*70}")
        
        # Benchmark without cache
        print(f"  Without KV-Cache...", end=" ", flush=True)
        try:
            _, time_no_cache = generate_without_cache(
                model, tokenizer, prompt, max_length=max_tokens, 
                device=device, temperature=temperature
            )
            print(f"✓ {time_no_cache:.2f}s")
        except Exception as e:
            print(f"✗ Error: {e}")
            time_no_cache = None
        
        # Benchmark with cache
        print(f"  With KV-Cache...", end=" ", flush=True)
        try:
            _, time_with_cache = generate_with_kv_cache(
                model, tokenizer, prompt, max_length=max_tokens, 
                device=device, temperature=temperature
            )
            print(f"✓ {time_with_cache:.2f}s")
        except Exception as e:
            print(f"✗ Error: {e}")
            time_with_cache = None
        
        # Calculate speedup
        if time_no_cache and time_with_cache and time_with_cache > 0:
            speedup = time_no_cache / time_with_cache
            time_saved = time_no_cache - time_with_cache
            results.append((max_tokens, time_no_cache, time_with_cache, speedup, time_saved))
        else:
            results.append((max_tokens, time_no_cache, time_with_cache, None, None))
    
    # Print summary table
    print("\n" + "="*70)
    print("Scaling Performance Summary")
    print("="*70)
    print(f"{'Tokens':<10} {'No Cache (s)':<15} {'With Cache (s)':<15} {'Speedup':<10} {'Time Saved (s)':<15}")
    print("-"*70)
    
    for max_tokens, time_no, time_with, speedup, time_saved in results:
        if speedup is not None:
            print(f"{max_tokens:<10} {time_no:<15.2f} {time_with:<15.2f} {speedup:<10.2f}x {time_saved:<15.2f}")
        else:
            print(f"{max_tokens:<10} {'Error':<15} {'Error':<15} {'-':<10} {'-':<15}")
    
    print("="*70)
    
    # Key insights
    if len(results) > 1 and all(r[3] is not None for r in results):
        print("\nScaling Insights:")
        first_speedup = results[0][3]
        last_speedup = results[-1][3]
        if first_speedup and last_speedup:
            print(f"- Speedup at {results[0][0]} tokens: {first_speedup:.2f}x")
            print(f"- Speedup at {results[-1][0]} tokens: {last_speedup:.2f}x")
            if last_speedup > first_speedup:
                improvement = ((last_speedup - first_speedup) / first_speedup) * 100
                print(f"- Speedup improved by {improvement:.1f}% as sequence length increased")
            print("- KV-Cache becomes more beneficial with longer sequences")
            print("- This demonstrates O(n²) → O(n) complexity reduction")
    
    # Plot chart if matplotlib is available
    if HAS_MATPLOTLIB and len(results) > 0:
        plot_scaling_results(results, model_name)


def plot_scaling_results(results: List[tuple], model_name: str, save_path: Optional[str] = None) -> None:
    """
    Plot scaling benchmark results as a chart.
    
    Args:
        results: List of (tokens, time_no_cache, time_with_cache, speedup, time_saved) tuples
        model_name: Model name for title
        save_path: Optional path to save the plot (if None, just display)
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Filter out errors
    valid_results = [r for r in results if r[3] is not None]
    if len(valid_results) == 0:
        print("\nNo valid results to plot.")
        return
    
    tokens = [r[0] for r in valid_results]
    time_no_cache = [r[1] for r in valid_results]
    time_with_cache = [r[2] for r in valid_results]
    speedups = [r[3] for r in valid_results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time vs Sequence Length
    ax1.plot(tokens, time_no_cache, 'o-', label='Without KV-Cache', linewidth=2, markersize=8, color='red')
    ax1.plot(tokens, time_with_cache, 's-', label='With KV-Cache', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax1.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax1.set_title(f'Generation Time vs Sequence Length\n{model_name}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(tokens)
    
    # Plot 2: Speedup vs Sequence Length
    ax2.plot(tokens, speedups, 'o-', label='Speedup', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax2.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax2.set_ylabel('Speedup (x times faster)', fontsize=12)
    ax2.set_title('KV-Cache Speedup vs Sequence Length', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(tokens)
    
    # Add annotations for speedup values
    for i, (token, speedup) in enumerate(zip(tokens, speedups)):
        ax2.annotate(f'{speedup:.2f}x', 
                    xy=(token, speedup), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
    else:
        plt.show()
    
    print("\nChart displayed. Close the window to continue.")


def main():
    """Example benchmark with Qwen 0.6B."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "The future of artificial intelligence"
    model_dir = "./models"  # Change to your preferred directory
    
    try:

        print("\n\n" + "="*70)
        print("\nScaling Benchmark (Multiple Lengths):")
        print("="*70)
        
        # Scaling benchmark
        benchmark_scaling(
            model_name=model_name,
            prompt=prompt,
            token_lengths=[10, 20, 50, 100, 200],
            model_dir=model_dir
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nInstall dependencies:")
        print("pip install transformers accelerate torch")


if __name__ == "__main__":
    main()

