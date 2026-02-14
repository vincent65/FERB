"""
Backend init/finalize for reference, triton, and numba_cuda.

Used by scripts/worker.py to set up and tear down the distributed environment
per backend.
"""

import os
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Backend: reference (NCCL)
# ---------------------------------------------------------------------------

def init_reference(rank: int, world_size: int) -> None:
    """Initialize torch.distributed for reference backend.

    - If CUDA is available: uses NCCL and binds each rank to a GPU.
    - If CUDA is NOT available (e.g. local Mac dev): uses GLOO on CPU so scripts can run.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                device_id=torch.device("cuda", rank),
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
    else:
        # CPU-only fallback for local development.
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )


def finalize_reference() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Backend: triton (nvshmem4py)
# ---------------------------------------------------------------------------

def init_triton(rank: int, world_size: int) -> None:
    """Initialize NVSHMEM via nvshmem4py (UID bootstrap). Call after torch.distributed init."""
    from cuda.core.experimental import Device
    import nvshmem.core as nvshmem

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()
    uniqueid = nvshmem.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()
    uniqueid = broadcast_objects[0]
    nvshmem.init(device=dev, uid=uniqueid, rank=rank_id, nranks=num_ranks, initializer_method="uid")


def finalize_triton() -> None:
    import nvshmem.core as nvshmem
    nvshmem.finalize()
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Backend: numba-cuda (nvshmem4py, UID bootstrap via torch.distributed)
# ---------------------------------------------------------------------------
# Uses UID bootstrap (like triton) so it works under torchrun; MPI bootstrap
# would require mpiexec and would see COMM_WORLD size 1 per process under torchrun.
# No global state - solutions use torch.cuda.current_stream() like triton.


### TODO: WE SHOULD NOT HAVE TO DO THIS!
def _patch_nvshmem_teardown() -> None:
    """Monkey-patch nvshmem4py's numba teardown callback to avoid TypeError on ObjectCode.

    nvshmem4py _numbast.py line 76 does: lambda x: nvshmem.bindings.cumodule_finalize(int(x))
    but numba-cuda passes an ObjectCode (not int-convertible). Wrap it to suppress the error.
    """
    try:
        import nvshmem.bindings.device.numba._numbast as _numbast
        _orig_register = getattr(_numbast, "register_shim", None)
        if _orig_register is None:
            return

        def _safe_register(shim_obj):
            result = _orig_register(shim_obj)
            orig_cb = getattr(shim_obj, "teardown_callback", None)
            if orig_cb is not None:
                def safe_cb(x):
                    try:
                        return orig_cb(x)
                    except (TypeError, ValueError):
                        pass  # suppress ObjectCode int() error on exit
                shim_obj.teardown_callback = safe_cb
            return result

        _numbast.register_shim = _safe_register
    except Exception:
        pass  # if the module structure changed, skip the patch


def init_numba_cuda(rank: int, world_size: int) -> None:
    """Initialize NVSHMEM via nvshmem4py (UID bootstrap). Call after torch.distributed init."""
    _patch_nvshmem_teardown()

    from cuda.core.experimental import Device
    import nvshmem.core as nvshmem

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()
    uniqueid = nvshmem.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()
    uniqueid = broadcast_objects[0]
    nvshmem.init(device=dev, uid=uniqueid, rank=rank_id, nranks=num_ranks, initializer_method="uid")


def finalize_numba_cuda() -> None:
    """Finalize NVSHMEM for numba-cuda backend."""
    import nvshmem.core as nvshmem
    nvshmem.finalize()
    if dist.is_initialized():
        dist.destroy_process_group()
