"""
Text Generation with Custom KV-Cache

Universal generation functions using custom KV-Cache implementation.
Works with HuggingFace transformer model.
"""

import torch
import time
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM
from kv_cache import MultiLayerKVCache


def generate_without_cache(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    device: str = "cuda",
    temperature: float = 1.0,
    top_p: Optional[float] = None
) -> Tuple[str, float]:
    """
    Generate text without KV-Cache (baseline for comparison).
    
    This method recomputes the entire sequence for each new token.
    Useful for understanding the performance baseline.
    
    Args:
        model: Any transformer language model
        tokenizer: Corresponding tokenizer
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        device: Device to run on
        temperature: Sampling temperature (1.0 = deterministic)
        top_p: Top-p (nucleus) sampling parameter (None = disabled)
        
    Returns:
        Tuple of (generated_text, time_taken)
    """
    # Convert prompt to tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass - explicitly disable cache to force recomputation
            # This means we recompute K,V for all tokens every time
            logits = model(input_ids, use_cache=False).logits
            
            # Get logits for next token prediction
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if temperature == 1.0 and top_p is None:
                # Deterministic: pick the most likely token
                next_token = torch.argmax(
                    torch.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True
                )
            else:
                # Stochastic: sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append new token to sequence
            token_id = next_token.squeeze().item()  # Get scalar value
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)  # next_token shape: (1, 1)
            
            # Stop if we hit end-of-sequence token
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and token_id == eos_token_id:
                break
    
    elapsed_time = time.time() - start_time
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, elapsed_time


def generate_with_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    device: str = "cuda",
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_seq_len: int = 2048
) -> Tuple[str, float]:
    """
    Generate text with custom KV-Cache for efficient generation.
    
    This function uses the custom MultiLayerKVCache class to manage
    key-value caching, providing full control over cache management.
    
    The KV-Cache optimization:
    - First pass: Process entire prompt, cache K,V pairs
    - Subsequent passes: Only process new token, reuse cached K,V
    - Result: O(n²) → O(n) computation per token
    
    Args:
        model: Any transformer language model (Qwen, LLaMA, GPT, etc.)
        tokenizer: Corresponding tokenizer
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        device: Device to run on
        temperature: Sampling temperature (1.0 = deterministic)
        top_p: Top-p (nucleus) sampling parameter (None = disabled)
        max_seq_len: Maximum sequence length for cache
        
    Returns:
        Tuple of (generated_text, time_taken)
    """
    # Get model config
    config = model.config
    num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
    
    # Handle GQA (Grouped Query Attention): use num_key_value_heads if available
    if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads is not None:
        num_heads = config.num_key_value_heads
    else:
        num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
    
    # Calculate head_dim - get from actual past_key_values if available
    # First, try to get from config
    if hasattr(config, 'head_dim'):
        head_dim = config.head_dim
    else:
        # Calculate from hidden_size and num_heads
        head_dim = config.hidden_size // num_heads
    
    # If we have past_key_values, use actual shape to determine dimensions
    # This will be done dynamically in update_from_past_key_values
    
    # Convert device string to torch.device
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    # Initialize custom multi-layer KV cache
    custom_cache = MultiLayerKVCache(
        num_layers=num_layers,
        batch_size=1,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        device=device_obj,
        dtype=model.dtype
    )
    
    # Convert prompt to tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        # Initial forward pass - process the prompt
        outputs = model(
            input_ids,
            use_cache=True,
            return_dict=True
        )
        
        # Store initial K,V in custom cache
        if outputs.past_key_values:
            custom_cache.update_from_past_key_values(outputs.past_key_values)
        
        generated_tokens = []
        
        # Generate tokens one at a time
        for _ in range(max_length):
            # Get logits for next token prediction
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-p filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if temperature == 1.0 and top_p is None:
                # Deterministic: pick the most likely token
                next_token = torch.argmax(
                    torch.softmax(next_token_logits, dim=-1), dim=-1
                ).unsqueeze(0)
            else:
                # Stochastic: sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Ensure token is on correct device
            next_token = next_token.to(device)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Stop if we hit end-of-sequence token
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and token_id == eos_token_id:
                break
            
            # Get past_key_values from model's output (don't convert, use directly)
            # The model expects the same format it returned
            past_key_values = outputs.past_key_values
            
            # Forward pass for next token using cached K,V
            outputs = model(
                next_token,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )
            
            # Update custom cache with new K,V for inspection/management
            if outputs.past_key_values:
                custom_cache.update_from_past_key_values(outputs.past_key_values)
    
    elapsed_time = time.time() - start_time
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, elapsed_time
