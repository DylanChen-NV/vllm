#!/usr/bin/env python3
import argparse
import py_compile
from pathlib import Path


HELPER = r'''


_mooncake_tcp_staging_cudart = None


def _mooncake_get_cudart():
    global _mooncake_tcp_staging_cudart
    if _mooncake_tcp_staging_cudart is None:
        library = ctypes.CDLL("libcudart.so.12")
        library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.cudaMemcpy.restype = ctypes.c_int
        _mooncake_tcp_staging_cudart = library
    return _mooncake_tcp_staging_cudart


def _mooncake_copy_segments_to_host(addrs, sizes):
    cudart = _mooncake_get_cudart()
    buffers = []
    host_addrs = []
    for row_addrs, row_sizes in zip(addrs, sizes, strict=True):
        row_buffers = []
        row_host_addrs = []
        for device_addr, size in zip(row_addrs, row_sizes, strict=True):
            buffer = ctypes.create_string_buffer(int(size))
            status = cudart.cudaMemcpy(
                ctypes.c_void_p(ctypes.addressof(buffer)),
                ctypes.c_void_p(int(device_addr)),
                ctypes.c_size_t(int(size)),
                2,
            )
            if status != 0:
                raise RuntimeError(f"Mooncake TCP staging D2H failed: cudaError={status}")
            row_buffers.append(buffer)
            row_host_addrs.append(ctypes.addressof(buffer))
        buffers.append(row_buffers)
        host_addrs.append(row_host_addrs)
    return buffers, host_addrs


def _mooncake_allocate_host_segments(sizes):
    buffers = []
    host_addrs = []
    for row_sizes in sizes:
        row_buffers = [ctypes.create_string_buffer(int(size)) for size in row_sizes]
        buffers.append(row_buffers)
        host_addrs.append([ctypes.addressof(buffer) for buffer in row_buffers])
    return buffers, host_addrs


def _mooncake_copy_segments_to_device(buffers, addrs, sizes, results):
    cudart = _mooncake_get_cudart()
    for row_index, (row_buffers, row_addrs, row_sizes) in enumerate(
        zip(buffers, addrs, sizes, strict=True)
    ):
        if row_index >= len(results) or int(results[row_index]) < 0:
            continue
        for buffer, device_addr, size in zip(
            row_buffers, row_addrs, row_sizes, strict=True
        ):
            status = cudart.cudaMemcpy(
                ctypes.c_void_p(int(device_addr)),
                ctypes.c_void_p(ctypes.addressof(buffer)),
                ctypes.c_size_t(int(size)),
                1,
            )
            if status != 0:
                raise RuntimeError(f"Mooncake TCP staging H2D failed: cudaError={status}")


def _mooncake_tcp_staged_batch_put(store, keys, addrs, sizes, replicate_config):
    buffers, host_addrs = _mooncake_copy_segments_to_host(addrs, sizes)
    results = store.batch_put_from_multi_buffers(
        keys, host_addrs, sizes, replicate_config
    )
    del buffers
    return results


def _mooncake_tcp_staged_batch_get(store, keys, addrs, sizes):
    buffers, host_addrs = _mooncake_allocate_host_segments(sizes)
    results = store.batch_get_into_multi_buffers(keys, host_addrs, sizes)
    _mooncake_copy_segments_to_device(buffers, addrs, sizes, results)
    return results
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/vllm"))
    args = parser.parse_args()
    path = args.root / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
    text = path.read_text()
    changed = False

    if "import ctypes\n" not in text:
        text = text.replace("import dataclasses\n", "import ctypes\nimport dataclasses\n", 1)
        changed = True
    if "def _mooncake_tcp_staged_batch_put(" not in text:
        marker = "logger = init_logger(__name__)"
        if marker not in text:
            raise RuntimeError("Mooncake TCP staging helper insertion target not found")
        text = text.replace(marker, marker + HELPER, 1)
        changed = True

    put_target = "res = self.store.batch_put_from_multi_buffers("
    if put_target in text:
        text = text.replace(
            put_target,
            "res = _mooncake_tcp_staged_batch_put(self.store,",
            1,
        )
        changed = True
    get_target = "res = self.store.batch_get_into_multi_buffers("
    if get_target in text:
        text = text.replace(
            get_target,
            "res = _mooncake_tcp_staged_batch_get(self.store,",
            1,
        )
        changed = True

    if changed:
        compile(text, str(path), "exec")
        path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    if "res = _mooncake_tcp_staged_batch_put(self.store," not in text:
        raise RuntimeError("Mooncake TCP staged put call was not installed")
    if "res = _mooncake_tcp_staged_batch_get(self.store," not in text:
        raise RuntimeError("Mooncake TCP staged get call was not installed")
    print(f"mooncake_tcp_cpu_staging_patch={changed}")


if __name__ == "__main__":
    main()
