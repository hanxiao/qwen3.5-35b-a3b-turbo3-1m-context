# Patches

## remove-slot-cap.patch

llama-server caps each slot's `n_ctx` to `n_ctx_train` (262K for Qwen3.5). This prevents using 1M context even when `--ctx-size 1048576` is specified. The patch comments out the capping line.

## hybrid-cache-reuse (PR #21099)

Qwen3.5 is a hybrid GDN + attention model. llama.cpp's KV cache reuse logic has a bug where `pos_min_thold` is set to `pos_next` when `n_swa=0`, causing every query to re-process the full context instead of prefix-matching.

The fix (4 changes in `src/llama-kv-cache-recurrent.cpp`):
1. `pos_min_thold = 0` for recurrent models
2. Checkpoint search uses position-matching semantics
3. `do_reset` preserves `n_past` instead of zeroing
4. Checkpoint erasure log cleanup

See: https://github.com/ggml-org/llama.cpp/pull/21099

If merged upstream, this patch is no longer needed.
