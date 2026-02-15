"""Generate synthetic demo data for iterations 6-10."""
import json
import os
import random

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "demo_problem4_optimization")
REF_MEAN_MS = 0.13808199763298035

def make_rank_data(rank, candidate_mean_ms, ref_mean_ms=REF_MEAN_MS):
    """Generate realistic per-rank timing data."""
    # Add slight per-rank variance
    jitter = random.uniform(0.92, 1.08)
    cand_ms = candidate_mean_ms * jitter
    ref_jitter = random.uniform(0.85, 1.15)
    ref_ms = ref_mean_ms * ref_jitter
    std_factor = random.uniform(0.05, 0.15)
    
    return {
        "rank": rank,
        "world_size": 8,
        "problem_id": 4,
        "backend": "agent",
        "backend_mode": "triton",
        "status": "ok",
        "correctness_ok": True,
        "correctness_note": "checked",
        "tolerances": {"rtol": 0.001, "atol": 0.001},
        "reference": None,
        "candidate": {
            "wall_time_ms": cand_ms,
            "wall_time_std_ms": cand_ms * std_factor,
            "min_time_ms": cand_ms * 0.85,
            "max_time_ms": cand_ms * 1.15,
            "iterations": 2,
        },
        "speedup_vs_ref": None,
        "reference_timing_skipped": True,
        "used_cached_reference_outputs": True,
        "trace_candidate": f"/logs/problem_4/agent/trace_candidate_rank{rank}.json",
    }

def make_reference_summary():
    """Reuse the same reference summary as the real run."""
    return {
        "problem_id": 4,
        "backend": "reference",
        "status": "ok",
        "n_ranks": 8,
        "ranks": [
            {
                "rank": r,
                "world_size": 8,
                "problem_id": 4,
                "backend": "reference",
                "backend_mode": "reference",
                "status": "ok",
                "correctness_ok": True,
                "correctness_note": "checked",
                "tolerances": {"rtol": 0.001, "atol": 0.001},
                "reference": {
                    "wall_time_ms": REF_MEAN_MS * random.uniform(0.85, 1.15),
                    "wall_time_std_ms": REF_MEAN_MS * 0.1,
                    "min_time_ms": REF_MEAN_MS * 0.8,
                    "max_time_ms": REF_MEAN_MS * 1.2,
                    "iterations": 2,
                },
                "candidate": {
                    "wall_time_ms": REF_MEAN_MS * random.uniform(0.85, 1.15),
                    "wall_time_std_ms": REF_MEAN_MS * 0.08,
                    "min_time_ms": REF_MEAN_MS * 0.78,
                    "max_time_ms": REF_MEAN_MS * 1.22,
                    "iterations": 2,
                },
                "speedup_vs_ref": random.uniform(1.1, 1.4),
                "reference_timing_skipped": False,
                "used_cached_reference_outputs": False,
            }
            for r in range(8)
        ],
        "aggregate": {
            "candidate_mean_ms": 0.11230799555778503,
            "candidate_p95_ms": 0.14352479577064514,
            "reference_mean_ms": REF_MEAN_MS,
            "reference_p95_ms": 0.16765840351581573,
            "speedup_vs_ref_mean": 1.229493919352643,
        },
    }


# Iteration 6: 0.15x speedup (still slow but improving)
# Iteration 7: 0.40x speedup
# Iteration 8: 0.72x speedup 
# Iteration 9: 1.02x speedup (matching reference)
# Iteration 10: 1.35x speedup (beating reference!)

ITER_CONFIGS = [
    {
        "iteration": 6,
        "candidate_mean_ms": REF_MEAN_MS / 0.15,  # ~0.92ms
        "score": 0.15,
        "event": "perf_opt_cycle",
    },
    {
        "iteration": 7,
        "candidate_mean_ms": REF_MEAN_MS / 0.40,  # ~0.345ms
        "score": 0.40,
        "event": "perf_opt_cycle",
    },
    {
        "iteration": 8,
        "candidate_mean_ms": REF_MEAN_MS / 0.72,  # ~0.192ms
        "score": 0.72,
        "event": "perf_opt_cycle",
    },
    {
        "iteration": 9,
        "candidate_mean_ms": REF_MEAN_MS / 1.02,  # ~0.135ms
        "score": 1.02,
        "event": "perf_opt_cycle",
    },
    {
        "iteration": 10,
        "candidate_mean_ms": REF_MEAN_MS / 1.35,  # ~0.102ms
        "score": 1.35,
        "event": "perf_opt_cycle",
    },
]

LLM_OUTPUTS = {
    6: {
        "diagnosis": [
            "The previous iteration timed out due to deadlock from redundant torch.cuda.synchronize() calls interfering with NVSHMEM barrier semantics",
            "Profiling traces show the barrier after zero_() was waiting indefinitely because cuda.synchronize() drained the stream before barrier could match",
            "BLOCK_SIZE=1024 may not be optimal for H100's SM architecture - need to explore larger blocks for better memory coalescing",
            "The atomic_add pattern is fundamentally correct (iter 3 proved correctness), but synchronization overhead dominates at ~12.7x slower"
        ],
        "hypotheses": [
            "Removing explicit torch.cuda.synchronize() and relying solely on NVSHMEM barriers will prevent the deadlock while maintaining correctness",
            "Increasing BLOCK_SIZE to 2048 will improve memory bandwidth utilization by reducing kernel launch grid size",
            "The zero-copy approach (removing unnecessary .clone() calls) will reduce memory allocation overhead",
            "Stream-ordered barrier semantics will properly synchronize without stalling the GPU pipeline"
        ],
        "test_expectations": [
            "Correctness should be maintained since we keep the proven atomic_add pattern from iter 3",
            "Performance should improve from 0.078x to ~0.15x by eliminating synchronization overhead",
            "No more timeout/deadlock since we removed the problematic synchronize-before-barrier pattern",
            "Memory bandwidth should be better utilized with BLOCK_SIZE=2048"
        ],
    },
    7: {
        "diagnosis": [
            "Iteration 6 succeeded at 0.15x speedup - synchronization fix worked, but still 6.7x slower than reference",
            "Profiling shows atomic_add contention on dst PE's buffer is the primary bottleneck - all 8 PEs writing to same addresses",
            "Simple BLOCK_SIZE increase to 2048 didn't help much because contention is per-address, not per-warp",
            "Memory access pattern analysis shows poor L2 cache utilization - blocks are too large for the cache line structure"
        ],
        "hypotheses": [
            "A tiled kernel approach with smaller TILE_SIZE will improve L2 cache hit rate by keeping working set within cache",
            "BLOCK_SIZE=4096 with TILE_SIZE=256 provides good balance between grid size and cache utilization",
            "Tiling also naturally reduces atomic contention since fewer PEs compete for the same cache lines simultaneously",
            "The fused tiled approach avoids separate kernel launch overhead per tile"
        ],
        "test_expectations": [
            "Performance should improve from 0.15x to ~0.4x through better cache utilization",
            "L2 cache hit rate should increase from ~30% to ~65% based on tile size analysis",
            "Atomic contention should decrease as tiles create temporal separation between PEs",
            "Correctness maintained since accumulation is still via atomic_add"
        ],
    },
    8: {
        "diagnosis": [
            "Iteration 7 achieved 0.40x speedup - tiling improved cache behavior significantly",
            "However, profiling reveals the tiled kernel itself has overhead: loop control, tile boundary checks add ~15% instruction overhead",
            "The per-tile atomic_add still creates contention hotspots within each 256-element tile",
            "NVLink trace analysis shows only ~45% of theoretical bandwidth being utilized"
        ],
        "hypotheses": [
            "Reverting to a simple single-pass kernel but with NVLink-tuned BLOCK_SIZE=1024 will eliminate tiling overhead",
            "H100 NVLink optimal transfer size is 128B (32 float32s) - BLOCK_SIZE=1024 = 32 warps = perfect alignment",
            "Ensuring contiguous float32 layout in memory enables 128-bit vectorized NVLink DMA transfers",
            "The in-place reshape on dst rank (instead of clone) avoids an extra O(N) memory copy",
            "Using reshape(-1) instead of flatten() avoids unnecessary contiguous copy when tensor is already contiguous"
        ],
        "test_expectations": [
            "Performance should improve from 0.40x to ~0.7x through NVLink bandwidth optimization",
            "NVLink utilization should increase from ~45% to ~65% with aligned block sizes",
            "Removing clone() on dst rank saves ~0.02ms per call for 1M element tensors",
            "Correctness maintained - same atomic_add semantics, just better memory layout"
        ],
    },
    9: {
        "diagnosis": [
            "Iteration 8 achieved 0.72x speedup - NVLink-tuned block size was a significant win",
            "Profiling shows NVLink utilization at ~62%, close to our target but still leaving bandwidth on the table",
            "Analysis of SM occupancy reveals BLOCK_SIZE=1024 creates 1024 concurrent threads per SM, exceeding register file capacity",
            "Register spills to local memory detected in nsight trace, adding ~0.03ms latency per kernel launch"
        ],
        "hypotheses": [
            "BLOCK_SIZE=512 will reduce register pressure and eliminate spills while maintaining good warp occupancy",
            "512 threads = 16 warps per SM is the sweet spot for H100's 128KB register file with our kernel's register usage",
            "Zero-copy input path (avoiding reshape when already float32 contiguous) saves allocation overhead",
            "In-place result extraction via view instead of clone reduces memory traffic by 50% on dst rank"
        ],
        "test_expectations": [
            "Performance should reach ~1.0x (matching reference) through optimal SM occupancy",
            "Register spills should be eliminated entirely with 512-thread blocks",
            "Total NVLink + compute efficiency should reach ~70% of theoretical peak",
            "Correctness maintained - atomic semantics unchanged"
        ],
    },
    10: {
        "diagnosis": [
            "Iteration 9 achieved 1.02x speedup - we matched the PyTorch/NCCL reference implementation!",
            "Profiling shows remaining overhead is in atomic_add contention at cache line granularity",
            "H100 L2 cache line size is 128B = 32 float32 values; BLOCK_SIZE=512 means 16 cache lines per block",
            "NVLink bandwidth utilization is at ~72% - there's still room to optimize cache line alignment"
        ],
        "hypotheses": [
            "BLOCK_SIZE=256 perfectly aligns to 1KB (8 cache lines of 128B), maximizing L2 efficiency",
            "Smaller blocks = more blocks = better load balancing across SMs when n_elements isn't perfectly divisible",
            "With 256-element blocks, each warp processes exactly 8 float32 values = half a cache line, reducing false sharing",
            "Zero-copy result path on dst rank (view of symmetric buffer directly) eliminates the last redundant copy",
            "Combined optimizations should push us past the reference to ~1.3-1.4x speedup"
        ],
        "test_expectations": [
            "Performance should exceed reference at ~1.35x speedup",
            "NVLink utilization should reach ~87% of theoretical 900 GB/s",
            "L2 cache hit rate should exceed 80% with cache-line-aligned blocks",
            "This represents the theoretical optimum for this atomic push-based approach on H100",
            "Correctness maintained throughout all optimizations"
        ],
    },
}


def main():
    random.seed(42)
    
    # --- Generate iterations.jsonl entries ---
    new_lines = []
    for cfg in ITER_CONFIGS:
        it = cfg["iteration"]
        cand_mean = cfg["candidate_mean_ms"]
        score = cfg["score"]

        ranks = [make_rank_data(r, cand_mean) for r in range(8)]
        actual_cand_mean = sum(r["candidate"]["wall_time_ms"] for r in ranks) / 8

        candidate_summary = {
            "problem_id": 4,
            "backend": "agent",
            "status": "ok",
            "n_ranks": 8,
            "ranks": ranks,
            "aggregate": {
                "candidate_mean_ms": actual_cand_mean,
                "candidate_p95_ms": actual_cand_mean * 1.05,
                "reference_mean_ms": None,
                "reference_p95_ms": None,
                "speedup_vs_ref_mean": None,
            },
        }

        reference_summary = make_reference_summary()

        candidate_feedback = {
            "status": "ok",
            "aggregate": candidate_summary["aggregate"],
            "correctness": {"all_ok": True, "failed_ranks": []},
            "traces": {
                "candidate": [f"/logs/problem_4/agent/trace_candidate_rank{r}.json" for r in range(8)],
                "reference": [],
            },
            "cache": {"used_cached_reference_outputs": True, "reference_timing_skipped": True},
        }

        reference_feedback = {
            "status": "ok",
            "aggregate": reference_summary["aggregate"],
            "correctness": {"all_ok": True, "failed_ranks": []},
            "traces": {
                "candidate": [f"/logs/problem_4/reference/trace_candidate_rank{r}.json" for r in range(8)],
                "reference": [f"/logs/problem_4/reference/trace_reference_rank{r}.json" for r in range(8)],
            },
            "cache": {"used_cached_reference_outputs": False, "reference_timing_skipped": False},
        }

        # Main evaluation event
        eval_event = {
            "iteration": it,
            "problem_id": 4,
            "reference_summary": reference_summary,
            "candidate_summary": candidate_summary,
            "reference_feedback": reference_feedback,
            "candidate_feedback": candidate_feedback,
            "score": score,
        }
        new_lines.append(json.dumps(eval_event))

        # Cycle event
        cycle_event = {"iteration": it, "event": cfg["event"], "problem_id": 4}
        new_lines.append(json.dumps(cycle_event))

        # Proposal applied event
        snap_path = os.path.abspath(os.path.join(DEMO_DIR, f"snapshots/problem_4/iter_{it}.py"))
        proposal_event = {
            "iteration": it,
            "event": "proposal_applied",
            "problem_id": 4,
            "snapshot": snap_path,
        }
        new_lines.append(json.dumps(proposal_event))

    # Append to iterations.jsonl
    iter_path = os.path.join(DEMO_DIR, "iterations.jsonl")
    with open(iter_path, "a") as f:
        for line in new_lines:
            f.write(line + "\n")

    print(f"Appended {len(new_lines)} lines to iterations.jsonl")

    # --- Generate LLM output JSON files ---
    llm_dir = os.path.join(DEMO_DIR, "llm_outputs")
    os.makedirs(llm_dir, exist_ok=True)

    for it, llm_data in LLM_OUTPUTS.items():
        snap_path = os.path.join(DEMO_DIR, f"snapshots/problem_4/iter_{it}.py")
        with open(snap_path) as f:
            code = f.read()

        output = {
            "event": "llm_proposal",
            "iteration": it,
            "problem_id": 4,
            "candidate_file": "/Users/vincentyip/programming/FERB/solutions_agent/4_agent.py",
            "proposal_mode": "perf_opt",
            "model": "claude-opus-4-6",
            "diagnosis": llm_data["diagnosis"],
            "hypotheses": llm_data["hypotheses"],
            "test_expectations": llm_data["test_expectations"],
            "candidate_code": code,
        }

        out_path = os.path.join(llm_dir, f"iter_{it}_problem_4.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {out_path}")

    # --- Update summary.json ---
    summary_path = os.path.join(DEMO_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "name": "demo_problem4_optimization",
            "run_dir": DEMO_DIR,
            "iterations": 10,
            "is_demo": True,
        }, f, indent=2)
    print("Updated summary.json")

    # --- Create demo_manifest.json for the simulation system ---
    manifest = {
        "is_demo": True,
        "total_iterations": 10,
        "delay_per_iteration_ms": 2500,
        "problem_ids": [4],
    }
    manifest_path = os.path.join(DEMO_DIR, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Created demo_manifest.json")

    print("\nDone! Demo data generated successfully.")


if __name__ == "__main__":
    main()
