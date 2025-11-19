import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_cache import PromptCacheManager
from generation import generate_with_prompt_cache

def main():
    # Setup
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

    cache_manager = PromptCacheManager(device=device)
    
    system_prompt = (
        "You are a helpful AI assistant that specializes in explaining complex scientific concepts "
        "to children. You should use simple analogies, avoid jargon, and be very encouraging. "
        "Always start your answer with 'Hey there, little explorer!' and end with 'Keep asking questions!'. "
        "Here is some context about the solar system: "
        "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. "
        "It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. "
        "The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. "
        "The four inner system planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, being composed primarily of rock and metal. "
        "The four giant planets of the outer system are substantially larger and more massive than the terrestrials. "
        "The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; "
        "the two outermost planets, Uranus and Neptune, are ice giants, being composed mostly of substances with relatively high melting points "
        "compared with hydrogen and helium, called volatiles, such as water, ammonia, and methane. "
        "There are also dwarf planets like Pluto. "
    ) * 5
    
    print(f"\nSystem Prompt Length: {len(system_prompt)} chars")
    
    print("\n--- Warmup: Caching System Prompt ---")
    _, _ = generate_with_prompt_cache(
        model, tokenizer, system_prompt, cache_manager, max_length=1, device=device
    )
    
    # Scenario 1: First Query (Should hit System Prompt cache)
    print("\n--- Run 1: First Query (Partial Cache Hit) ---")
    user_query_1 = "Why is the sky blue?"
    full_prompt_1 = f"{system_prompt}\nUser: {user_query_1}\nAssistant:"
    
    response_1, time_1 = generate_with_prompt_cache(
        model, tokenizer, full_prompt_1, cache_manager, max_length=50, device=device
    )
    print(f"Response: {response_1}")
    print(f"Time: {time_1:.4f}s")
    
    # Scenario 2: Second run (Should hit System Prompt cache again)
    # We use the SAME system prompt, but a different user query.
    # The cache manager should match the system prompt prefix.
    print("\n--- Run 2: Second Query (Partial Cache Hit) ---")
    user_query_2 = "Tell me about Mars."
    full_prompt_2 = f"{system_prompt}\nUser: {user_query_2}\nAssistant:"
    
    response_2, time_2 = generate_with_prompt_cache(
        model, tokenizer, full_prompt_2, cache_manager, max_length=50, device=device
    )
    print(f"Response: {response_2}")
    print(f"Time: {time_2:.4f}s")
    
    print(f"\nSpeedup (Run 2 vs Run 1): {time_1 / time_2:.2f}x (Should be similar if both hit cache)")
    
    # To see the REAL speedup, we should compare with a run that DOES NOT use cache at all (or clear cache)
    print("\n--- Run 3: Baseline (No Cache) ---")
    cache_manager.clear()
    response_3, time_3 = generate_with_prompt_cache(
        model, tokenizer, full_prompt_2, cache_manager, max_length=50, device=device
    )
    print(f"Time: {time_3:.4f}s")
    
    print(f"\nTrue Speedup (Cache vs No Cache): {time_3 / time_2:.2f}x")

if __name__ == "__main__":
    main()
