"""Correctness-gated dense matmul benchmark for triton_kernels."""

import argparse
import json
import re
from pathlib import Path

import torch
import triton
from triton.compiler import CompiledKernel
from triton_kernels.matmul import PrecisionConfig, matmul, specializations
from triton_kernels.tensor import make_ragged_tensor_metadata

DTYPE_CHOICES = ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2", "int8")
SHAPE_SUITES = {
    "square": ((256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096)),
    "h3": (
        (9322, 21504, 5376),
        (9322, 5376, 7168),
        (9322, 28672, 5376),
        (9322, 5376, 14336),
    ),
    "rectangular": ((4096, 256, 4096), (256, 4096, 4096)),
    "small-m": ((1, 4096, 4096), (16, 4096, 4096), (64, 8192, 4096)),
    "irregular": ((257, 509, 263), (1000, 704, 800), (9322, 5377, 7169)),
}


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = re.split(r"[xX,]", value)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K (for example 9322,5376,7168)")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness-gated dense triton_kernels.matmul benchmark with config and ISA reporting.")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="Additional M,N,K shape. May be repeated.",
    )
    parser.add_argument(
        "--suite",
        choices=("custom", *SHAPE_SUITES, "all"),
        default="custom",
        help="Named shape bucket. custom uses --m/--n/--k; --shape entries are always appended.",
    )
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--semantic-mode",
        choices=("dense", "openai_ragged"),
        default="dense",
        help="dense uses plain 2D inputs; openai_ragged matches the single-expert ragged OpenAI harness.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("cuda_graph", "eager"),
        default="cuda_graph",
        help="Timing method. cuda_graph uses triton.testing.do_bench_cudagraph.",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup time in ms for eager timing only.")
    parser.add_argument("--rep", type=int, default=100, help="Rep time in ms for cuda_graph or eager timing.")
    parser.add_argument(
        "--transpose-w",
        choices=("both", "false", "true"),
        default="both",
        help="Which W layout(s) to benchmark. true uses transpose-contiguous-transpose storage.",
    )
    parser.add_argument("--correctness-only", action="store_true", help="Compile and validate without timing.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the complete run record as JSON.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Write final AMDGCN and TTGIR for each selected matmul kernel.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Use Proton for timing and emit .hatchet profiles into this directory.",
    )
    parser.add_argument("--profile-reps", type=int, default=100)
    return parser.parse_args()


def resolve_shapes(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    shapes = []
    if args.suite == "custom":
        shapes.append((args.m, args.n, args.k))
    elif args.suite == "all":
        for suite in SHAPE_SUITES.values():
            shapes.extend(suite)
    else:
        shapes.extend(SHAPE_SUITES[args.suite])
    shapes.extend(args.shape)
    return list(dict.fromkeys(shapes))


def make_random(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dtype == torch.int8:
        # Negative values are required to verify gfx12 signedness control bits.
        return torch.randint(-8, 8, shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)


def make_weight(k: int, n: int, dtype: torch.dtype, device: torch.device, transpose_w: bool) -> torch.Tensor:
    if transpose_w:
        return make_random((n, k), dtype, device).transpose(-1, -2)
    return make_random((k, n), dtype, device).contiguous()


def make_batched_weight(k: int, n: int, dtype: torch.dtype, device: torch.device, transpose_w: bool) -> torch.Tensor:
    if transpose_w:
        return make_random((1, n, k), dtype, device).transpose(-1, -2)
    return make_random((1, k, n), dtype, device).contiguous()


def output_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.int8:
        return torch.int32
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float32
    return dtype


def make_inputs(
    *,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    transpose_w: bool,
    semantic_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict, tuple[int, ...]]:
    if semantic_mode == "dense":
        a = make_random((m, k), dtype, device)
        b = make_weight(k, n, dtype, device, transpose_w)
        return a, b, None, {}, tuple(b.stride())

    slice_sizes = torch.tensor([m], dtype=torch.int32, device=device)
    a_ragged_metadata = make_ragged_tensor_metadata(slice_sizes, m)
    a = make_random((m, k), dtype, device)
    b = make_batched_weight(k, n, dtype, device, transpose_w)
    bias = torch.randn((1, n), device=device, dtype=torch.float32)
    return a, b, bias, {"a_ragged_metadata": a_ragged_metadata}, tuple(b.stride())


def make_precision_config(dtype: torch.dtype, semantic_mode: str) -> PrecisionConfig:
    return PrecisionConfig(
        out_dtype=output_dtype(dtype),
        intermediate_out_dtype=torch.int32 if dtype == torch.int8 else None,
        # Match the existing OpenAI harness for its fp16/bf16 path.
        flexpoint_saturate_inf=semantic_mode == "openai_ragged",
    )


def check_capability(dtype: torch.dtype, semantic_mode: str) -> None:
    target = triton.runtime.driver.active.get_current_target()
    arch = str(target.arch)
    if dtype == torch.int8:
        if semantic_mode != "dense":
            raise RuntimeError("INT8 benchmark supports only --semantic-mode=dense")
        if target.backend != "hip" or not arch.startswith("gfx12") or arch.startswith("gfx125"):
            raise RuntimeError("INT8 triton_kernels.matmul currently requires an AMD RDNA4 target")
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if target.backend == "cuda" and torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("FP8 benchmark requires Hopper or newer on CUDA")
        if target.backend == "hip" and not (arch == "gfx950" or arch.startswith("gfx12")):
            raise RuntimeError("OCP FP8 benchmark requires gfx950 or gfx12 on HIP")


def reference_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
    semantic_mode: str,
) -> torch.Tensor:
    b_dense = b[0] if semantic_mode == "openai_ragged" else b
    # INT8 test values are bounded so FP32 represents every product and partial sum exactly.
    if dtype == torch.int8 and a.shape[-1] * 64 >= 2**24:
        raise RuntimeError("INT8 correctness reference requires K * 64 < 2**24")
    reference = a.float() @ b_dense.float()
    if bias is not None:
        reference += bias[0]
    return reference.to(output_dtype(dtype))


def check_correctness(actual: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype) -> dict:
    if dtype == torch.int8:
        correct = torch.equal(actual, reference)
        max_abs = int((actual - reference).abs().max().item())
        if not correct:
            raise AssertionError(f"INT8 correctness failed with max_abs={max_abs}")
        return {"passed": True, "rtol": 0.0, "atol": 0.0, "max_abs": max_abs, "max_rel": 0.0}

    rtol, atol = {
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float8_e4m3fn: (2e-2, 0.5),
        torch.float8_e5m2: (2e-2, 0.5),
    }[dtype]
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    diff = (actual_f32 - reference_f32).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / reference_f32.abs().clamp_min(1e-12)).max().item())
    torch.testing.assert_close(actual_f32, reference_f32, rtol=rtol, atol=atol)
    return {"passed": True, "rtol": rtol, "atol": atol, "max_abs": max_abs, "max_rel": max_rel}


def capture_main_launch(run) -> tuple[torch.Tensor, CompiledKernel]:
    launches = []

    def hook(lazy_metadata) -> None:
        metadata = lazy_metadata.get()
        name = metadata.get("name", "")
        if name.startswith(("_matmul", "_p_matmul")):
            device = triton.runtime.driver.active.get_current_device()
            launches.append((name.split(" [", 1)[0], metadata["function"], device))

    triton.knobs.runtime.launch_enter_hook.add(hook)
    try:
        output = run()
        torch.cuda.synchronize()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(hook)
    if not launches:
        raise RuntimeError("No triton_kernels matmul launch was captured")
    return output, find_compiled_kernel(*launches[-1])


def find_compiled_kernel(name: str, function: int, device: int) -> CompiledKernel:
    kernels = specializations.get()
    for attr in ("_matmul", "_p_matmul"):
        jit_kernel = getattr(kernels, attr)
        device_cache = jit_kernel.device_caches.get(device)
        if device_cache is None:
            continue
        for compiled in device_cache[0].values():
            # Names omit alignment, EVEN_K, and other specialization properties.
            # The loaded function handle identifies the kernel that actually ran.
            if compiled.name == name and getattr(compiled, "function", None) == function:
                return compiled
    raise RuntimeError(
        f"Compiled kernel {name!r} with function {function} on device {device} was not found in the JIT cache")


def extract_kernel_record(compiled) -> dict:
    metadata = compiled.metadata
    amdgcn = compiled.asm.get("amdgcn", "")
    block_match = re.search(r"_(\d+)x(\d+)x(\d+)x(\d+)$", compiled.name)
    if block_match is None:
        raise RuntimeError(f"Could not parse matmul config from {compiled.name}")
    block_m, block_n, block_k, split_k = (int(value) for value in block_match.groups())
    matrix_instructions = sorted(set(re.findall(r"\bv_(?:wmma|mfma)[^\s,]*", amdgcn)))
    private_matches = re.findall(r"(?:\.amdhsa_private_segment_fixed_size|\.private_segment_fixed_size:)\s*(\d+)",
                                 amdgcn)
    sgpr_matches = re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", amdgcn)
    return {
        "name": compiled.name,
        "hash": compiled.hash,
        "config": {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "split_k": split_k,
            "num_warps": metadata.num_warps,
            "num_stages": metadata.num_stages,
            "matrix_instr_nonkdim": getattr(metadata, "matrix_instr_nonkdim", None),
            "kpack": getattr(metadata, "kpack", None),
            "waves_per_eu": getattr(metadata, "waves_per_eu", None),
        },
        "resources": {
            "vgpr": compiled.n_regs,
            "sgpr": int(sgpr_matches[-1]) if sgpr_matches else None,
            "lds_bytes": metadata.shared,
            "private_segment_bytes": int(private_matches[-1]) if private_matches else None,
            "spills": compiled.n_spills,
        },
        "matrix_instructions": matrix_instructions,
        "matrix_instruction_count": sum(amdgcn.count(instruction) for instruction in matrix_instructions),
    }


def write_kernel_artifacts(compiled, directory: Path, stem: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for asm_key in ("amdgcn", "ttgir"):
        value = compiled.asm.get(asm_key)
        if not isinstance(value, str):
            continue
        path = directory / f"{stem}.{asm_key}"
        path.write_text(value)
        paths[asm_key] = str(path)
    return paths


def percentile(samples: list[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(samples, dtype=torch.float64), q).item())


def coefficient_of_variation(samples: list[float]) -> float:
    values = torch.tensor(samples, dtype=torch.float64)
    mean = values.mean()
    return float((values.std(correction=0) / mean).item()) if mean != 0 else 0.0


def measure(run, benchmark_mode: str, warmup: int, rep: int) -> dict:
    if benchmark_mode == "cuda_graph":
        samples = triton.testing.do_bench_cudagraph(run, rep=rep, return_mode="all")
    else:
        samples = triton.testing.do_bench(run, warmup=warmup, rep=rep, return_mode="all")
    samples = [float(value) for value in samples]
    return {
        "samples_ms": samples,
        "p20_ms": percentile(samples, 0.2),
        "median_ms": percentile(samples, 0.5),
        "p80_ms": percentile(samples, 0.8),
        "cv": coefficient_of_variation(samples),
    }


def read_proton_matmul_avg_ms(profile_path: Path) -> float:
    profile = json.loads(profile_path.read_text())
    candidates = []

    def collect(node: dict) -> None:
        frame = node.get("frame", {})
        metrics = node.get("metrics", {})
        if "_p_matmul" in frame.get("name", "") or "_matmul" in frame.get("name", ""):
            candidates.append(metrics)
        for child in node.get("children", []):
            collect(child)

    for node in profile:
        collect(node)
    if not candidates:
        raise RuntimeError(f"No matmul kernel found in Proton profile {profile_path}")
    metrics = max(candidates, key=lambda item: max((v for k, v in item.items() if k.startswith("flops")), default=0))
    return metrics["time (ns)"] / metrics["count"] / 1e6


def measure_proton(run, profile_path: Path, profile_reps: int) -> dict:
    if profile_reps <= 0:
        raise ValueError("--profile-reps must be positive")
    import triton.profiler as proton

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    proton.start(str(profile_path), hook="triton")
    for _ in range(profile_reps):
        run()
    torch.cuda.synchronize()
    proton.finalize()
    ms = read_proton_matmul_avg_ms(profile_path.with_suffix(".hatchet"))
    return {"samples_ms": [ms], "p20_ms": ms, "median_ms": ms, "p80_ms": ms, "cv": 0.0}


def benchmark_case(
    *,
    case_index: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    transpose_w: bool,
    args: argparse.Namespace,
) -> dict:
    a, b, bias, matmul_kwargs, b_stride = make_inputs(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        device=device,
        transpose_w=transpose_w,
        semantic_mode=args.semantic_mode,
    )
    precision_config = make_precision_config(dtype, args.semantic_mode)
    output = torch.empty((m, n), device=device, dtype=output_dtype(dtype))

    def run() -> torch.Tensor:
        return matmul(a, b, bias, c=output, precision_config=precision_config, **matmul_kwargs)

    actual, compiled = capture_main_launch(run)
    reference = reference_matmul(a, b, bias, dtype, args.semantic_mode)
    correctness = check_correctness(actual, reference, dtype)
    del reference

    kernel_record = extract_kernel_record(compiled)
    dtype_name = str(dtype).removeprefix("torch.")
    layout_name = "transpose_contiguous_transpose" if transpose_w else "contiguous"
    stem = f"{case_index:03d}_{dtype_name}_{m}x{n}x{k}_{layout_name}"
    artifacts = write_kernel_artifacts(compiled, args.artifact_dir, stem) if args.artifact_dir else {}

    timing = None
    if not args.correctness_only:
        if args.profile_dir is None:
            timing = measure(run, args.benchmark_mode, args.warmup, args.rep)
        else:
            timing = measure_proton(run, args.profile_dir / stem, args.profile_reps)
        timing["throughput_tops"] = 2.0 * m * n * k / (timing["median_ms"] * 1e9)

    return {
        "shape": {"m": m, "n": n, "k": k},
        "dtype": dtype_name,
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "semantic_mode": args.semantic_mode,
        "transpose_w": transpose_w,
        "b_stride": list(b_stride),
        "correctness": correctness,
        "timing": timing,
        "kernel": kernel_record,
        "artifacts": artifacts,
    }


def iter_transpose_values(choice: str) -> list[bool]:
    if choice == "both":
        return [False, True]
    return [choice == "true"]


def target_record(device: torch.device) -> dict:
    target = triton.runtime.driver.active.get_current_target()
    props = torch.cuda.get_device_properties(device)
    pci_bus_id = None
    if all(hasattr(props, name) for name in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        pci_bus_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    return {
        "backend": target.backend,
        "arch": str(target.arch),
        "warp_size": target.warp_size,
        "device_index": device.index,
        "device_name": torch.cuda.get_device_name(device),
        "uuid": str(getattr(props, "uuid", "")) or None,
        "pci_bus_id": pci_bus_id,
    }


def print_result(result: dict) -> None:
    shape = result["shape"]
    config = result["kernel"]["config"]
    resources = result["kernel"]["resources"]
    timing = result["timing"]
    timing_text = "correctness-only"
    if timing is not None:
        timing_text = (f"p20/median/p80={timing['p20_ms']:.3f}/{timing['median_ms']:.3f}/{timing['p80_ms']:.3f} ms "
                       f"{timing['throughput_tops']:.3f} TOPS")
    print(f"M={shape['m']} N={shape['n']} K={shape['k']} transpose_w={result['transpose_w']} "
          f"correct=True {timing_text}\n"
          f"  config=BM{config['block_m']}xBN{config['block_n']}xBK{config['block_k']} "
          f"warps={config['num_warps']} stages={config['num_stages']} split_k={config['split_k']}\n"
          f"  resources=VGPR{resources['vgpr']} SGPR{resources['sgpr']} LDS={resources['lds_bytes']}B "
          f"private={resources['private_segment_bytes']}B spills={resources['spills']} "
          f"matrix={result['kernel']['matrix_instructions']}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = getattr(torch, args.dtype)
    check_capability(dtype, args.semantic_mode)
    shapes = resolve_shapes(args)
    target = target_record(device)
    measurement_mode = "proton" if args.profile_dir is not None else args.benchmark_mode

    print(f"target={target['backend']}:{target['arch']} device={target['device_name']} dtype={args.dtype} "
          f"suite={args.suite} semantic_mode={args.semantic_mode} measurement_mode={measurement_mode}")
    results = []
    for shape in shapes:
        for transpose_w in iter_transpose_values(args.transpose_w):
            result = benchmark_case(
                case_index=len(results),
                m=shape[0],
                n=shape[1],
                k=shape[2],
                dtype=dtype,
                device=device,
                transpose_w=transpose_w,
                args=args,
            )
            results.append(result)
            print_result(result)

    record = {
        "schema_version": 1,
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "target": target,
        "seed": args.seed,
        "suite": args.suite,
        "measurement_mode": measurement_mode,
        "parameters": {
            "dtype": args.dtype,
            "semantic_mode": args.semantic_mode,
            "transpose_w": args.transpose_w,
            "benchmark_mode": args.benchmark_mode,
            "warmup_ms": args.warmup,
            "rep_ms": args.rep,
            "profile_reps": args.profile_reps if args.profile_dir is not None else None,
            "correctness_only": args.correctness_only,
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"JSON written to {args.output_json}")


if __name__ == "__main__":
    main()
