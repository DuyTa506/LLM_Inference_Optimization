import torch
from typing import Dict, Tuple, Optional, List
from kv_cache import KVCache, MultiLayerKVCache

class PromptCacheManager:
    """
    Manages KV-Caches for different prompts to enable Prompt Caching.
    
    This allows reusing the computation of shared prefixes (e.g., system prompts),
    significantly reducing Time To First Token (TTFT).
    """
    
    def __init__(self, device: str = "cuda", max_cache_size: int = 10):
        """
        Initialize PromptCacheManager.
        
        Args:
            device: Device to store caches on
            max_cache_size: Maximum number of prompts to cache
        """
        self.device = device
        self.max_cache_size = max_cache_size
        # Dictionary to store caches: prompt_text -> MultiLayerKVCache
        self.caches: Dict[str, MultiLayerKVCache] = {}
        # Keep track of usage for simple LRU eviction if needed (not implemented in v1)
        self.access_history: List[str] = []
    
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from kv_cache import MultiLayerKVCache

class RadixNode:
    """
    A node in the Radix Tree.
    """
    def __init__(self):
        self.children: Dict[int, 'RadixNode'] = {}  # Map token_id -> RadixNode
        self.kv_cache: Optional[MultiLayerKVCache] = None
        self.last_accessed: float = time.time()
        self.access_count: int = 0
        self.parent: Optional['RadixNode'] = None
        self.token_id: Optional[int] = None # The token ID that leads to this node

class RadixCache:
    """
    Radix Tree (Trie) based cache for KV states.
    Allows O(L) prefix matching and efficient memory sharing.
    """
    def __init__(self, device: str = "cuda", max_cache_size: int = 100):
        self.root = RadixNode()
        self.device = device
        self.max_cache_size = max_cache_size # Limit on number of cached nodes with KV states
        self.current_cache_size = 0
        
    def insert(self, token_ids: List[int], kv_cache: MultiLayerKVCache) -> None:
        """
        Insert a KV cache for a sequence of tokens.
        """
        node = self.root
        for token in token_ids:
            if token not in node.children:
                new_node = RadixNode()
                new_node.parent = node
                new_node.token_id = token
                node.children[token] = new_node
            node = node.children[token]
        
        # Store the cache at the leaf (or internal node corresponding to the sequence)
        if node.kv_cache is None:
            self.current_cache_size += 1
            
        node.kv_cache = kv_cache
        node.last_accessed = time.time()
        node.access_count += 1
        
        # Evict if needed
        if self.current_cache_size > self.max_cache_size:
            self.evict()

    def match_prefix(self, token_ids: List[int]) -> Tuple[Optional[MultiLayerKVCache], int]:
        """
        Find the longest matching prefix in the tree.
        If the matching node doesn't have a cache, try to find one in its descendants
        and use it (the caller will truncate).
        
        Returns:
            Tuple[Optional[MultiLayerKVCache], int]: (Cached KV state, Number of matched tokens)
        """
        node = self.root
        matched_len = 0
        
        # Traverse to find the longest matching path
        for i, token in enumerate(token_ids):
            if token in node.children:
                node = node.children[token]
                matched_len += 1
            else:
                break
        
        # If the node has a cache, return it
        if node.kv_cache is not None:
            node.last_accessed = time.time()
            node.access_count += 1
            return node.kv_cache, matched_len
            
        # If not, search descendants for any cache
        # We perform a BFS/DFS to find the nearest cache
        candidate_node = self._find_first_descendant_with_cache(node)
        if candidate_node:
            # We found a cache in a descendant!
            # We return this cache, but the matched_len remains the length of the prefix match.
            # The caller MUST truncate this cache to matched_len.
            candidate_node.last_accessed = time.time()
            candidate_node.access_count += 1
            return candidate_node.kv_cache, matched_len
            
        # Fallback: Check ancestors if we overshot (unlikely with this logic, 
        # but if we matched a path that has no cache anywhere, we might want to back up.
        # However, our logic above stops at the first mismatch. 
        # If the path up to there has no cache and no descendants have cache, 
        # we might check ancestors. But usually we want the longest match.)
        
        # Actually, we should check ancestors if the current branch has no cache.
        # But the loop above goes down. If we stop at a node with no cache and no descendants with cache,
        # we should backtrack to find the longest prefix that *does* have a cache (or can reach one).
        # But for now, let's stick to the descendant strategy as it solves the "shared prefix" case.
        
        return None, 0

    def _find_first_descendant_with_cache(self, node: RadixNode) -> Optional[RadixNode]:
        """Helper to find a descendant with a cache."""
        # BFS to find nearest
        queue = [node]
        while queue:
            curr = queue.pop(0)
            if curr.kv_cache is not None:
                return curr
            queue.extend(curr.children.values())
        return None

    def evict(self, num_to_evict: int = 10) -> None:
        """
        Evict least recently used nodes.
        """
        # Collect all nodes with caches
        nodes_with_cache = []
        
        def traverse(node):
            if node.kv_cache is not None:
                nodes_with_cache.append(node)
            for child in node.children.values():
                traverse(child)
        
        traverse(self.root)
        
        # Sort by last accessed time (ascending)
        nodes_with_cache.sort(key=lambda x: x.last_accessed)
        
        # Remove caches from the oldest ones
        for i in range(min(num_to_evict, len(nodes_with_cache))):
            node = nodes_with_cache[i]
            node.kv_cache = None # Free the memory
            self.current_cache_size -= 1
            
            # Optional: Prune leaf nodes that have no cache and no children
            # This is a bit complex to do safely while iterating, so skipping for now
            # as the main memory cost is the KV tensors, not the tree nodes.

    def clear(self) -> None:
        """Clear the entire cache."""
        self.root = RadixNode()
        self.current_cache_size = 0

# Alias for backward compatibility if needed, or we can update calls
PromptCacheManager = RadixCache
