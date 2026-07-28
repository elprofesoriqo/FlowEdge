# Loader

Reads a `.safetensors` file with no dependencies.

1. mmap the file.
2. Parse the JSON header with a hand-written scan.
3. Copy each tensor into the arena as F32, widening BF16 on the way.
4. Fill a fixed table of `TensorView`, each holding a name, shape, and pointer.

The header is read once at load, and every offset is bounds-checked.

## BF16 to F32

BF16 is the top half of an IEEE-754 float: 1 sign bit, 8 exponent bits, 7 mantissa bits. It keeps the exponent range of F32, so widening is a left shift, exact and lossless:

$$
\text{f32bits} = \text{bf16bits} \ll 16
$$

The engine runs in F32, so the loader pays this once at load and the runtime never sees BF16.

## Why this way

Loading is a data-movement problem, so the design targets page residency and layout, not parse speed.

mmap maps the file into the address space without a read syscall per tensor. But a mapped page is not resident until it is first touched, and a first touch during a step is a minor fault that blows the latency budget. The copy into the arena is the pre-fault: it walks every weight once at load, so the pages are warm before the control loop starts.

The copy also fixes layout. File offsets are not 64-byte aligned, and the stored dtype may be BF16. Moving the tensor into the arena makes it 64-byte aligned and single-dtype, so a kernel loads F32 without a per-element convert.

Widening to F32 doubles the weight bytes, and that trade is now in question. It was made to keep the hot path convert-free, but `matmul` turns out to be limited by weight bandwidth, so the extra bytes cost more than the convert would. See [ADR 0007](../decisions/0007-roofline).

The header is a flat map of name to dtype, shape, and byte offsets, so a hundred-line scan reads it. A JSON library would add a dependency, binary size, and a supply-chain surface for a grammar this small.

Source: `src/loader/safetensors.h` and `safetensors.cc`.
