"""
Performance measurement utilities for ParallelKernelBench.

This module provides functions to measure:
1. Wall-clock time (using CUDA events for accurate GPU timing)
2. Bandwidth utilization (calculated from bytes transferred / time)
"""

import torch
import time
from typing import Dict, Tuple, Optional


def calculate_bytes_transferred(
    problem_id: int,
    tensor_shape: tuple,
    dtype: torch.dtype,
    world_size: int
) -> int:
    """
    Calculate the total bytes transferred for a given collective operation.
    
    This is an approximation - actual bytes depend on the algorithm used.
    For simplicity, we calculate the theoretical minimum bytes for each operation.
    
    Args:
        problem_id: Problem ID (1=allreduce, 2=allgather, etc.)
        tensor_shape: Shape of the input tensor
        dtype: Data type
        world_size: Number of ranks/GPUs
        
    Returns:
        Total bytes transferred (aggregate across all ranks)
    """
    element_size = torch.tensor(0, dtype=dtype).element_size()
    # Calculate number of elements from shape tuple
    n_elements = 1
    for dim in tensor_shape:
        n_elements *= dim
    tensor_bytes = n_elements * element_size
    
    # Theoretical bytes transferred for each collective operation
    # These are lower bounds - actual implementations may transfer more
    if problem_id == 1:  # All-reduce (sum)
        # Each rank sends its data and receives the result
        # Ring algorithm: ~2 * (n_pes - 1) * tensor_bytes per rank
        # Total: world_size * 2 * (world_size - 1) * tensor_bytes
        return world_size * 2 * (world_size - 1) * tensor_bytes
    
    elif problem_id == 2:  # All-gather
        # Each rank sends its data to all others
        # Total: world_size * (world_size - 1) * tensor_bytes
        return world_size * (world_size - 1) * tensor_bytes
    
    elif problem_id == 3:  # Broadcast
        # Root sends to all others
        # Total: (world_size - 1) * tensor_bytes
        return (world_size - 1) * tensor_bytes
    
    elif problem_id == 4:  # Reduce
        # All ranks send to root
        # Total: (world_size - 1) * tensor_bytes
        return (world_size - 1) * tensor_bytes
    
    elif problem_id == 5:  # Scatter
        # Root sends chunks to all others
        # Each chunk is tensor_bytes / world_size
        chunk_bytes = tensor_bytes // world_size
        return (world_size - 1) * chunk_bytes
    
    elif problem_id == 6:  # Gather
        # All ranks send to root
        # Total: (world_size - 1) * tensor_bytes
        return (world_size - 1) * tensor_bytes
    
    elif problem_id == 7:  # Reduce-scatter
        # Each rank sends its data and receives a chunk
        # Similar to all-reduce but with scatter
        chunk_bytes = tensor_bytes // world_size
        return world_size * 2 * (world_size - 1) * chunk_bytes
    
    elif problem_id == 8:  # All-to-all
        # Each rank sends different data to every other rank
        # Input shape is [world_size, *chunk_shape]
        chunk_bytes = tensor_bytes // world_size
        return world_size * (world_size - 1) * chunk_bytes
    
    elif problem_id == 9:
        # LayerNorm backward: Two all-reduces (d_beta and d_gamma)
        # Input shape is [B, H] but communication is on [H] vectors
        # Extract H dimension: if shape is (B, H), then H = shape[1]
        # vector_bytes = H * element_size = tensor_bytes / B
        if len(tensor_shape) >= 2:
            H = tensor_shape[1]  # Hidden dimension
            element_size = torch.tensor(0, dtype=dtype).element_size()
            vector_bytes = H * element_size
        else:
            # Fallback: assume tensor_bytes is already the vector size
            vector_bytes = tensor_bytes
        # Two all-reduces, each with world_size * 2 * (world_size - 1) * vector_bytes
        return 2 * world_size * 2 * (world_size - 1) * vector_bytes
    
    else:
        # Default: assume all-reduce
        return world_size * 2 * (world_size - 1) * tensor_bytes


def measure_solution_performance(
    solution_fn,
    tensor: torch.Tensor,
    problem_id: int,
    world_size: int,
    warmup_iters: int = 3,
    measure_iters: int = 10,
    use_cuda_events: bool = True
) -> Dict[str, float]:
    """
    Measure wall-clock time and bandwidth for a solution function.
    
    Args:
        solution_fn: The solution function to measure (takes tensor, returns tensor)
        tensor: Input tensor
        problem_id: Problem ID for bandwidth calculation
        world_size: Number of ranks/GPUs
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
        use_cuda_events: Use CUDA events for timing (more accurate than Python time)
        
        Returns:
        Dictionary with performance metrics:
        - 'wall_time_ms': Average wall-clock time in milliseconds (this rank's GPU time)
        - 'wall_time_std_ms': Standard deviation of wall-clock time
        - 'bandwidth_gbps': Theoretical aggregate bandwidth if this rank's time was the bottleneck
                           (calculated as total_bytes / this_rank's_time)
                           NOTE: Different ranks will have slightly different values because
                           they measure slightly different times. The true aggregate bandwidth
                           should use max_time across all ranks (see compare_performance.py).
        - 'bandwidth_std_gbps': Standard deviation of bandwidth
        - 'min_time_ms': Minimum time
        - 'max_time_ms': Maximum time
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = solution_fn(tensor)
        torch.cuda.synchronize()
    
    # Measurement
    times_ms = []
    
    if use_cuda_events:
        # Use CUDA events for accurate GPU timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
        
        for i in range(measure_iters):
            start_events[i].record()
            _ = solution_fn(tensor)
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        for i in range(measure_iters):
            elapsed_ms = start_events[i].elapsed_time(end_events[i])
            times_ms.append(elapsed_ms)
    else:
        # Fallback to Python time (less accurate)
        for _ in range(measure_iters):
            start = time.perf_counter()
            _ = solution_fn(tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
    
    # Calculate statistics
    import numpy as np
    times_ms = np.array(times_ms)
    
    avg_time_ms = float(np.mean(times_ms))
    std_time_ms = float(np.std(times_ms))
    min_time_ms = float(np.min(times_ms))
    max_time_ms = float(np.max(times_ms))
    
    # Calculate bandwidth
    bytes_transferred = calculate_bytes_transferred(
        problem_id, tensor.shape, tensor.dtype, world_size
    )
    
    # Convert bytes to GB and time to seconds
    bytes_per_iter = bytes_transferred
    avg_time_s = avg_time_ms / 1000.0
    
    # Calculate bandwidth: total_bytes / this_rank's_time
    # NOTE: This gives slightly different bandwidth per rank because each rank
    # measures slightly different times. The actual operation finishes when the
    # slowest rank finishes, so the "true" aggregate bandwidth should use max_time
    # across all ranks. This per-rank bandwidth is a "what-if" calculation:
    # "If this rank's time was the bottleneck, what would the bandwidth be?"
    bandwidth_gbps = (bytes_per_iter / (1024**3)) / avg_time_s
    bandwidth_std_gbps = (bytes_per_iter / (1024**3)) / (std_time_ms / 1000.0) if std_time_ms > 0 else 0.0
    
    return {
        'wall_time_ms': avg_time_ms,
        'wall_time_std_ms': std_time_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'bandwidth_std_gbps': bandwidth_std_gbps,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms,
        'bytes_transferred': bytes_transferred,
        'iterations': measure_iters,
    }


def format_performance_report(metrics: Dict[str, float], rank: int = 0) -> str:
    """Format performance metrics as a human-readable string."""
    lines = [
        f"[Rank {rank}] Performance Metrics:",
        f"  Wall-clock time: {metrics['wall_time_ms']:.3f} ± {metrics['wall_time_std_ms']:.3f} ms",
        f"  Time range: [{metrics['min_time_ms']:.3f}, {metrics['max_time_ms']:.3f}] ms",
        f"  Bandwidth: {metrics['bandwidth_gbps']:.2f} ± {metrics['bandwidth_std_gbps']:.2f} GB/s",
        f"  Bytes transferred: {metrics['bytes_transferred'] / (1024**2):.2f} MB",
        f"  Iterations: {metrics['iterations']}",
    ]
    return "\n".join(lines)

