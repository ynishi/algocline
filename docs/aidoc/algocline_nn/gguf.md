# algocline-nn::gguf

Writing a model out as GGUF.

Everything this crate trains lands in safetensors, which is the
format the training side reads and writes and nothing else does.
llama.cpp, Ollama and LM Studio read GGUF, so a model trained here
could be evaluated here and nowhere else.

This is the writer. It renames the tensors to the
[GGUF naming convention](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
llama.cpp expects, emits the architecture's key-value metadata,
optionally embeds the tokenizer, and quantizes on the way out.

# What is verified here, and what is not

Verified by this crate's tests: the file is a GGUF that reads back,
every tensor survives the round trip with its GGUF name and its
shape, the metadata says what it was asked to say, and a quantized
export is within the error its block format allows.

**Not verified here: that llama.cpp loads it.** That takes llama.cpp,
which this repository does not build or vendor. The naming and the
key set below are written against its documented convention, and the
remaining step is one `llama-cli -m <file>` on a machine that has it.
Said plainly rather than implied, because "exports GGUF" and "runs
under llama.cpp" are different claims and only the first is fenced.

# Layout

ggml stores a matmul weight with the input dimension first. A candle
[`candle_nn::Linear`] holds `[out, in]`, and candle's GGUF writer
emits dimensions reversed, so passing these tensors through
unchanged writes `ne = [in, out]` — the convention ggml reads. The
transposition is the writer's, not ours, which is why nothing here
transposes anything. (candle's reader reverses them again, so a
round trip through this crate comes back in candle order; the file
on disk is in ggml's.)

# Quantization has a width requirement

Every ggml block format stores a fixed number of values together —
32 for the `Q*_0` / `Q*_1` families, 256 for the K-quants — and the
last dimension of a tensor has to be a multiple of it. A model whose
hidden size is not is exportable at F32 or F16 and not below; the
refusal names the tensor and the block size rather than failing
somewhere inside candle.

## Functions

- `export_gguf` — Write `varmap` to `path` as GGUF.
- `export_gguf_with_metadata` — [`export_gguf`], with `extra` key-value entries written beside the
- `parse_precision` — Parse the wire form of a precision (Lua opts, JSON config).

## Types

- `GgufArch` — Which architecture's naming and key set to write.
- `GgufReport` — What an export wrote.
- `GgufSpec` — The facts a reader needs to rebuild the computation.

## Constants

- `PRECISION_NAMES` — Every precision name this version accepts, widest first.

