import torch
import time
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, DynamicCache
from kv_cache import MultiLayerKVCache
from prompt_cache import PromptCacheManager

def generate_with_prompt_cache(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    cache_manager: PromptCacheManager,
    max_length: int = 100,
    device: str = "cuda",
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_seq_len: int = 2048
) -> Tuple[str, float]:
    """
    Generate text using Prompt Caching to speed up inference.
    
    Logic:
    1. Check if a prefix of the prompt exists in cache_manager.
    2. If found, reuse the KV cache for that prefix.
    3. Process only the new suffix tokens.
    4. Continue generation.
    5. Store the updated cache for the full prompt (optional, but good for future).
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Full input prompt
        cache_manager: The PromptCacheManager instance
        ... (standard generation args)
        
    Returns:
        Tuple of (generated_text, time_taken)
    """
    # Get model config details
    config = model.config
    num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
    
    if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads is not None:
        num_heads = config.num_key_value_heads
    else:
        num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        
    if hasattr(config, 'head_dim'):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // num_heads

    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device

    start_time = time.time()
    
    # 1. Tokenize the full prompt first
    full_tokens_tensor = tokenizer.encode(prompt, return_tensors='pt').to(device_obj)
    full_tokens_list = full_tokens_tensor[0].tolist()
    
    # 2. Query the Radix Tree for the longest prefix match
    cached_kv, matched_len = cache_manager.match_prefix(full_tokens_list)
    
    input_ids = full_tokens_tensor
    past_key_values = None
    current_cache = None
    
    if cached_kv is not None and matched_len > 0:
        # Cache Hit Logic
        print(f"Cache Hit! Reusing {matched_len} tokens.")
        
        # Create a fresh cache instance
        current_cache = MultiLayerKVCache(
            num_layers=cached_kv.num_layers,
            batch_size=cached_kv.batch_size,
            num_heads=cached_kv.num_heads,
            head_dim=cached_kv.head_dim,
            max_seq_len=cached_kv.max_seq_len,
            device=device_obj,
            dtype=cached_kv.dtype
        )
        
        # Copy state from cached_kv
        current_cache.update_from_past_key_values(cached_kv.to_past_key_values())
        
        # Truncate if necessary
        if matched_len < len(current_cache):
             current_cache.truncate(matched_len)
        
        # Special Case: If we matched the ENTIRE prompt, we have no tokens left to process to generate logits.
        # We must backtrack by 1 token so that we can feed the last token to the model
        # to generate the logits for the *next* (new) token.
        if matched_len == len(full_tokens_list):
            print("Full cache match! Backtracking by 1 token to generate logits.")
            matched_len -= 1
            current_cache.truncate(matched_len)
            
        # Prepare input for the model (suffix only)
        input_ids = full_tokens_tensor[:, matched_len:]
        
        # Get past_key_values for the model
        past_key_values = DynamicCache.from_legacy_cache(current_cache.to_past_key_values())
        
    else:
        # Cache Miss
        print("No cache hit. Processing full prompt.")
        current_cache = MultiLayerKVCache(
            num_layers=num_layers,
            batch_size=1,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            device=device_obj,
            dtype=model.dtype
        )
        past_key_values = None

    # Generation Loop
    with torch.no_grad():
        # If we have input_ids (suffix or full prompt), process them
        if input_ids.shape[1] > 0:
            outputs = model(
                input_ids,
                use_cache=True,
                past_key_values=past_key_values, # Use cached KV if available
                return_dict=True
            )
            
            # Update cache with the result of processing input_ids
            if outputs.past_key_values:
                # Convert DynamicCache back to tuple if needed
                if hasattr(outputs.past_key_values, 'to_legacy_cache'):
                    new_past_key_values = outputs.past_key_values.to_legacy_cache()
                else:
                    new_past_key_values = outputs.past_key_values
                
                current_cache.update_from_past_key_values(new_past_key_values)
            
            # Prepare for next token generation
            next_token_logits = outputs.logits[:, -1, :] / temperature
        else:
            # Edge case: prompt was exactly the cached prefix
            # We need to generate the next token based on the cache
            # For simplicity, we assume we always have at least one new token or we re-feed the last token of prefix.
            pass

        generated_tokens = []
        next_token = None
        
        # Handle the case where input_ids was empty (full match)
        if input_ids.shape[1] == 0:
             # This implies we matched the entire prompt.
             pass

        for _ in range(max_length):
            # Sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            if temperature == 1.0 and top_p is None:
                next_token = torch.argmax(torch.softmax(next_token_logits, dim=-1), dim=-1).unsqueeze(0)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            next_token = next_token.to(device_obj)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            
            # Get current KV from cache (which is now up to date)
            past_key_values_tuple = current_cache.to_past_key_values()
            # Convert to DynamicCache
            past_key_values = DynamicCache.from_legacy_cache(past_key_values_tuple)
            
            outputs = model(
                next_token,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )
            
            if outputs.past_key_values:
                # Convert DynamicCache back to tuple if needed
                if hasattr(outputs.past_key_values, 'to_legacy_cache'):
                    new_past_key_values = outputs.past_key_values.to_legacy_cache()
                else:
                    new_past_key_values = outputs.past_key_values
                
                current_cache.update_from_past_key_values(new_past_key_values)
                
            next_token_logits = outputs.logits[:, -1, :] / temperature

    elapsed_time = time.time() - start_time
    
    # Store the final state in the cache
    prompt_len = len(full_tokens_list)
    
    # Create a storage copy
    storage_cache = MultiLayerKVCache(
        num_layers=current_cache.num_layers,
        batch_size=current_cache.batch_size,
        num_heads=current_cache.num_heads,
        head_dim=current_cache.head_dim,
        max_seq_len=current_cache.max_seq_len,
        device=device_obj,
        dtype=current_cache.dtype
    )
    storage_cache.update_from_past_key_values(current_cache.to_past_key_values())
    storage_cache.truncate(prompt_len)
    
    cache_manager.insert(full_tokens_list, storage_cache)
    print(f"Stored cache for prompt (Size: {prompt_len} tokens)")
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, elapsed_time
