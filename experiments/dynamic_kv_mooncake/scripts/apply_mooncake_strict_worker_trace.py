#!/usr/bin/env python3
import argparse
import py_compile
from pathlib import Path


HELPER = '''


def _mooncake_strict_key_digests(keys: list[str]) -> list[str]:
    return [f"{idx}:{hashlib.sha1(str(key).encode('utf-8')).hexdigest()[:16]}" for idx, key in enumerate(keys)]


def _mooncake_strict_success_keys(keys: list[str], results: Any) -> list[str]:
    try:
        result_values = list(results)
    except Exception:
        return list(keys)
    output: list[str] = []
    for key, value in zip(keys, result_values, strict=False):
        try:
            ok = int(value) >= 0
        except Exception:
            ok = bool(value)
        if ok:
            output.append(key)
    return output


def _mooncake_strict_event(phase: str, keys: list[str], *, success: bool = True, **extra: Any) -> None:
    if os.getenv("MOONCAKE_STRICT_REUSE_TRACE", "0") != "1":
        return
    try:
        node_ip = os.getenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME") or get_ip()
    except Exception:
        node_ip = os.getenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "unknown")
    payload: dict[str, Any] = {
        "backend": "Mooncake",
        "wall_ns": time.time_ns(),
        "phase": phase,
        "node_ip": node_ip,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "request_id": str(extra.pop("request_id", "unknown")),
        "success": bool(success),
        "key_count": int(len(keys)),
        "key_digests": _mooncake_strict_key_digests(keys),
        "verl_replica_rank": os.getenv("MOONCAKE_VERL_REPLICA_RANK", "unknown"),
        "verl_node_rank": os.getenv("MOONCAKE_VERL_NODE_RANK", "unknown"),
    }
    for key, value in extra.items():
        payload[key] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    logger.warning("MOONCAKE_STRICT_REUSE_EVENT " + json.dumps(payload, sort_keys=True))
'''

SEND_EVENT = '''                success_keys_for_event = _mooncake_strict_success_keys([str(key) for key in keys], res)
                _mooncake_strict_event(
                    "FINISHED_SEND", success_keys_for_event,
                    success=bool(success_keys_for_event), request_id=req_id,
                    batch_keys=len(keys), success_keys=len(success_keys_for_event),
                    failed_keys=max(0, len(keys) - len(success_keys_for_event)),
                    token_len=int(token_len),
                )
'''

RECV_EVENT = '''                success_keys_for_event = _mooncake_strict_success_keys([str(key) for key in batch_keys], res)
                _mooncake_strict_event(
                    "FINISHED_RECV", success_keys_for_event,
                    success=bool(success_keys_for_event), request_id=req_id,
                    batch_keys=len(batch_keys), success_keys=len(success_keys_for_event),
                    failed_keys=max(0, len(batch_keys) - len(success_keys_for_event)),
                    token_len=int(token_len),
                    vllm_cached_tokens=int(req_meta.load_spec.vllm_cached_tokens),
                    kvpool_cached_tokens=int(req_meta.load_spec.kvpool_cached_tokens),
                )
'''


def insert_after_call(text: str, target: str, insertion: str) -> tuple[str, bool]:
    if insertion.strip() in text:
        return text, False
    start = text.find(target)
    if start < 0:
        raise RuntimeError(f"call target not found: {target}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                line_end = text.find("\n", index)
                line_end = len(text) if line_end < 0 else line_end + 1
                return text[:line_end] + insertion + text[line_end:], True
    raise RuntimeError(f"unterminated call target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/vllm"))
    args = parser.parse_args()
    path = args.root / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
    text = path.read_text()
    changed = False
    strict_info = 'logger.info("MOONCAKE_STRICT_REUSE_EVENT "'
    if strict_info in text:
        text = text.replace(
            strict_info, 'logger.warning("MOONCAKE_STRICT_REUSE_EVENT "', 1
        )
        changed = True
    if "import hashlib\n" not in text:
        text = text.replace("import dataclasses\n", "import dataclasses\nimport hashlib\n", 1)
        changed = True
    if "import time\n" not in text:
        text = text.replace("import socket\n", "import socket\nimport time\n", 1)
        changed = True
    wall_field = "        \"wall_ns\": time.time_ns(),\n"
    if wall_field not in text:
        text = text.replace(
            "        \"backend\": \"Mooncake\",\n",
            "        \"backend\": \"Mooncake\",\n" + wall_field,
            1,
        )
        changed = True
    if "MOONCAKE_STRICT_REUSE_EVENT" not in text:
        marker = "logger = init_logger(__name__)\n\nDEFAULT_GLOBAL_SEGMENT_SIZE"
        if marker not in text:
            raise RuntimeError("worker helper insertion target not found")
        text = text.replace(marker, "logger = init_logger(__name__)" + HELPER + "\n\nDEFAULT_GLOBAL_SEGMENT_SIZE", 1)
        changed = True
    if "self._mooncake_cuda_device_index" not in text:
        init_target = "        self._record_operation_cb = record_operation\n"
        init_replacement = init_target + (
            "        # CUDA current device is thread-local; transfer threads must restore the worker device.\n"
            "        self._mooncake_cuda_device_index = (\n"
            "            torch.cuda.current_device() if torch.cuda.is_available() else None\n"
            "        )\n"
        )
        run_target = "    def run(self):\n        self.ready_event.set()\n"
        run_replacement = (
            "    def run(self):\n"
            "        if self._mooncake_cuda_device_index is not None:\n"
            "            torch.cuda.set_device(self._mooncake_cuda_device_index)\n"
            "        self.ready_event.set()\n"
        )
        if init_target not in text or run_target not in text:
            raise RuntimeError("KVTransferThread CUDA device insertion target not found")
        text = text.replace(init_target, init_replacement, 1)
        text = text.replace(run_target, run_replacement, 1)
        changed = True
    if "mooncake_store_setup_cuda_device" not in text:
        setup_target = "        # Initialize MooncakeDistributedStore with its own TransferEngine\n"
        setup_replacement = setup_target + (
            "        # Mooncake creates CUDA-using C++ worker threads during setup. Bind the\n"
            "        # process to this vLLM worker's GPU before those threads are created.\n"
            "        mooncake_store_setup_cuda_device = None\n"
            "        if torch.cuda.is_available():\n"
            "            mooncake_store_setup_cuda_device = (\n"
            "                parallel_config.rank % torch.cuda.device_count()\n"
            "            )\n"
            "            torch.cuda.set_device(mooncake_store_setup_cuda_device)\n"
            "        logger.info(\n"
            "            'Mooncake store setup CUDA device: rank=%d tp_rank=%d device=%s',\n"
            "            parallel_config.rank, self.tp_rank, mooncake_store_setup_cuda_device,\n"
            "        )\n"
        )
        if setup_target not in text:
            raise RuntimeError("Mooncake store setup CUDA device insertion target not found")
        text = text.replace(setup_target, setup_replacement, 1)
        changed = True
    text, did_change = insert_after_call(text, "res = self.store.batch_put_from_multi_buffers(", SEND_EVENT)
    changed |= did_change
    text, did_change = insert_after_call(text, "res = self.store.batch_get_into_multi_buffers(", RECV_EVENT)
    changed |= did_change
    if changed:
        compile(text, str(path), "exec")
        path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    print(f"mooncake_strict_worker_trace_patch={changed}")


if __name__ == "__main__":
    main()
