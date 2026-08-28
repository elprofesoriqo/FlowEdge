# Loader

Reads a `.safetensors` file with no dependencies.

1. mmap the file.
2. Parse the JSON header with a hand-written scan.
3. Copy each tensor into the arena while preserving its F32 or BF16 representation.
4. Fill a fixed table of `TensorView`, each holding a name, shape, and pointer.

The header is read once at load, and every offset is bounds-checked.

## BF16 storage

BF16 is the top half of an IEEE-754 float: 1 sign bit, 8 exponent bits, 7 mantissa bits. It keeps the exponent range of F32, so widening is a left shift, exact and lossless:

$$
\text{f32bits} = \text{bf16bits} \ll 16
$$

The loader preserves BF16 weights to halve memory traffic. Matmul widens them directly in its
SIMD register path; small vectors and activations remain F32.

## Why this way

Loading is a data-movement problem, so the design targets page residency and layout, not parse speed.

mmap maps the file into the address space without a read syscall per tensor. But a mapped page is not resident until it is first touched, and a first touch during a step is a minor fault that blows the latency budget. The copy into the arena is the pre-fault: it walks every weight once at load, so the pages are warm before the control loop starts.

The copy also fixes layout. File offsets are not 64-byte aligned. Moving a tensor into the arena
makes it 64-byte aligned, so kernels can load it predictably.

BF16 is widened inline only where a matmul consumes it. That keeps the long-lived weight
footprint compact while preserving F32 activations. See [ADR 0009](../decisions/0009-bf16-weights).

The header is a flat map of name to dtype, shape, and byte offsets, so a hundred-line scan reads it. A JSON library would add a dependency, binary size, and a supply-chain surface for a grammar this small.

Source: `src/loader/safetensors.h` and `safetensors.cc`.
