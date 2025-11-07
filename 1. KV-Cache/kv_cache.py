"""
KV-Cache Implementation from Scratch

KV-Cache is an optimization technique for transformer models that caches
the key and value tensors from previous tokens during autoregressive generation.
This avoids recomputing attention for previously processed tokens.

Reference: https://apetulante.github.io/posts/KV-Caching/kv_caching.html
"""

import torch
from typing import Optional, Tuple


class KVCache:
    """
    Key-Value Cache for efficient autoregressive generation.
    
    During generation, we cache the computed keys and values for all previous
    tokens. For each new token, we only compute attention with the new token
    and append to the cache, rather than recomputing for all tokens.
    
    This reduces computation from O(n²) to O(n) per token during generation.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize KV-Cache.
        
        Args:
            batch_size: Batch size for the cache
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length to cache
            device: Device to store cache on (default: CPU)
            dtype: Data type for cache tensors
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Initialize empty cache tensors
        # Shape: (batch_size, num_heads, max_seq_len, head_dim) (B x H x L x D)
        self.k_cache = torch.zeros(
            (batch_size, num_heads, max_seq_len, head_dim),
            device=self.device,
            dtype=self.dtype
        )
        self.v_cache = torch.zeros(
            (batch_size, num_heads, max_seq_len, head_dim),
            device=self.device,
            dtype=self.dtype
        )
        
        # Track current sequence length in cache
        self.current_len = 0
    
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Update cache with new keys and values.
        
        This appends the new K,V pairs to the existing cache, allowing
        incremental generation without recomputing previous tokens.
        
        Args:
            keys: New key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            values: New value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = keys.size(2)
        
        # Check if we exceed max sequence length
        if self.current_len + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {self.current_len + seq_len} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
        
        # Append new keys and values to cache at current position
        self.k_cache[:, :, self.current_len:self.current_len + seq_len] = keys
        self.v_cache[:, :, self.current_len:self.current_len + seq_len] = values
        
        # Update current length
        self.current_len += seq_len
    
    def get(self, start_pos: int = 0, end_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached keys and values.
        
        Args:
            start_pos: Start position in cache (default: 0)
            end_pos: End position in cache (default: current_len)
        
        Returns:
            Tuple of (keys, values) tensors
        """
        if end_pos is None:
            end_pos = self.current_len
        
        # Slice cache tensors to get requested range
        k = self.k_cache[:, :, start_pos:end_pos]
        v = self.v_cache[:, :, start_pos:end_pos]
        
        return k, v
    
    def reset(self) -> None:
        """Reset cache to empty state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0
    
    def __len__(self) -> int:
        """Return current cached sequence length."""
        return self.current_len


class MultiLayerKVCache:
    """
    Multi-layer KV-Cache wrapper for HuggingFace transformer models.
    
    Manages separate KVCache instances for each transformer layer,
    allowing custom cache management across all layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize multi-layer KV-Cache.
        
        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size for the cache
            num_heads: Number of attention heads per layer (initial estimate, may be updated)
            head_dim: Dimension of each attention head (initial estimate, may be updated)
            max_seq_len: Maximum sequence length to cache
            device: Device to store cache on
            dtype: Data type for cache tensors
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Initialize caches lazily - will be created with correct dimensions on first update
        self.caches = None
        # Store original format from HuggingFace for proper conversion
        self.original_format = None  # Will store format info
    
    def update_from_past_key_values(self, past_key_values: Tuple) -> None:
        """
        Update caches from HuggingFace past_key_values format.
        
        This replaces the entire cache with the new past_key_values,
        since HuggingFace's past_key_values already contains all cached tokens.
        
        Args:
            past_key_values: Tuple of (k, v) tuples, one per layer
                Format: ((k1, v1), (k2, v2), ..., (kn, vn))
        """
        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {len(past_key_values)}"
            )
        
        # Detect format and dimensions from first layer
        if past_key_values and len(past_key_values) > 0:
            k_first, _ = past_key_values[0]
            
            # Detect actual dimensions from tensor
            if k_first.dim() == 3:
                # 3D tensor: (num_heads, seq_len, head_dim) or (num_heads, head_dim, seq_len)
                actual_num_heads = k_first.size(0)
                if k_first.size(1) >= k_first.size(2):
                    # (num_heads, seq_len, head_dim)
                    self.original_format = '3d_seq_last'
                    actual_head_dim = k_first.size(2)
                else:
                    # (num_heads, head_dim, seq_len)
                    self.original_format = '3d_head_last'
                    actual_head_dim = k_first.size(1)
            elif k_first.dim() == 4:
                # 4D tensor: (batch, num_heads, seq_len, head_dim) vs (batch, num_heads, head_dim, seq_len)
                actual_num_heads = k_first.size(1)
                if k_first.size(2) >= k_first.size(3):
                    # (batch, num_heads, seq_len, head_dim) - standard format
                    self.original_format = '4d_seq_last'
                    actual_head_dim = k_first.size(3)
                else:
                    # (batch, num_heads, head_dim, seq_len) - transpose needed
                    self.original_format = '4d_head_last'
                    actual_head_dim = k_first.size(2)
            else:
                self.original_format = '4d_seq_last'  # Default assumption
                actual_num_heads = self.num_heads
                actual_head_dim = self.head_dim
            
            # Update dimensions if different from initial estimate
            if actual_num_heads != self.num_heads or actual_head_dim != self.head_dim:
                self.num_heads = actual_num_heads
                self.head_dim = actual_head_dim
                # Recreate caches with correct dimensions
                self.caches = [
                    KVCache(self.batch_size, self.num_heads, self.head_dim, self.max_seq_len, self.device, self.dtype)
                    for _ in range(self.num_layers)
                ]
            elif self.caches is None:
                # Initialize caches if not already created
                self.caches = [
                    KVCache(self.batch_size, self.num_heads, self.head_dim, self.max_seq_len, self.device, self.dtype)
                    for _ in range(self.num_layers)
                ]
        
        # Ensure caches are initialized
        if self.caches is None:
            self.caches = [
                KVCache(self.batch_size, self.num_heads, self.head_dim, self.max_seq_len, self.device, self.dtype)
                for _ in range(self.num_layers)
            ]
        
        # Reset all caches first
        self.reset()
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Convert to our internal format: (batch, num_heads, seq_len, head_dim)
            if k.dim() == 3:
                # Add batch dimension: (num_heads, seq_len, head_dim) -> (1, num_heads, seq_len, head_dim)
                if self.original_format == '3d_head_last':
                    # (num_heads, head_dim, seq_len) -> transpose -> (num_heads, seq_len, head_dim)
                    k = k.transpose(-2, -1)
                    v = v.transpose(-2, -1)
                # Add batch dimension
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            elif k.dim() == 4:
                if self.original_format == '4d_head_last':
                    # Transpose from (batch, num_heads, head_dim, seq_len) to (batch, num_heads, seq_len, head_dim)
                    k = k.transpose(-2, -1)
                    v = v.transpose(-2, -1)
                # Now k, v should be (batch, num_heads, seq_len, head_dim)
            
            # Update cache (this will append, but since we reset, it's a fresh start)
            self.caches[layer_idx].update(k, v)
    
    def to_past_key_values(self) -> Tuple:
        """
        Convert caches to HuggingFace past_key_values format.
        
        Returns:
            Tuple of (k, v) tuples in HuggingFace format (same format as original)
        """
        result = []
        for cache in self.caches:
            k, v = cache.get()  # Shape: (batch, num_heads, seq_len, head_dim)
            
            # Convert back to original format
            if self.original_format == '3d_seq_last':
                # Remove batch dimension: (1, num_heads, seq_len, head_dim) -> (num_heads, seq_len, head_dim)
                k = k.squeeze(0)
                v = v.squeeze(0)
            elif self.original_format == '3d_head_last':
                # Remove batch and transpose: (1, num_heads, seq_len, head_dim) -> (num_heads, head_dim, seq_len)
                k = k.squeeze(0).transpose(-2, -1)
                v = v.squeeze(0).transpose(-2, -1)
            elif self.original_format == '4d_head_last':
                # Transpose back to (batch, num_heads, head_dim, seq_len)
                k = k.transpose(-2, -1)
                v = v.transpose(-2, -1)
            # else: '4d_seq_last' - keep as is (batch, num_heads, seq_len, head_dim)
            
            result.append((k, v))
        
        return tuple(result)
    
    def reset(self) -> None:
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()
    
    def __len__(self) -> int:
        """Return current cached sequence length (same for all layers)."""
        return len(self.caches[0]) if self.caches else 0

