"""
Simple Example: Using Custom KV-Cache with Any Model

This example shows how to use the custom KV-Cache implementation with any transformer model.
Works with Qwen, LLaMA, GPT, or any other HuggingFace model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation import generate_without_cache, generate_with_kv_cache


def main():
    """Example usage with any model."""
    # You can use any model here: Qwen, LLaMA, GPT, etc.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Change to any model you want
    
    # Specify model directory (models will be downloaded here)
    model_dir = "./models"  # Change to your preferred directory
    
    print(f"Loading model: {model_name}")
    print(f"Model directory: {model_dir}")
    print("(First time will download the model, may take a few minutes...)\n")
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load model and tokenizer (will download to model_dir if not already present)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=model_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=model_dir
        )
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {device}\n")
        
        # Example prompt
        prompt = "The future of artificial intelligence"
        print(f"Prompt: {prompt}\n")
        
        # Generate without cache (baseline)
        print("1. Generating WITHOUT KV-Cache...")
        result_no_cache, time_no_cache = generate_without_cache(
            model, tokenizer, prompt, max_length=30, device=device, temperature=0.8
        )
        print(f"   Time: {time_no_cache:.2f}s")
        print(f"   Result: {result_no_cache[:80]}...\n")
        
        # Generate with Custom KV-Cache (optimized)
        print("2. Generating WITH Custom KV-Cache...")
        result_with_cache, time_with_cache = generate_with_kv_cache(
            model, tokenizer, prompt, max_length=30, device=device, temperature=0.8
        )
        print(f"   Time: {time_with_cache:.2f}s")
        print(f"   Result: {result_with_cache[:80]}...\n")
        
        # Compare performance
        if time_with_cache > 0:
            speedup = time_no_cache / time_with_cache
            print("="*60)
            print(f"Speedup: {speedup:.2f}x faster with KV-Cache")
            print(f"Time saved: {time_no_cache - time_with_cache:.2f}s")
            print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nInstall dependencies:")
        print("pip install transformers accelerate torch")


if __name__ == "__main__":
    main()

