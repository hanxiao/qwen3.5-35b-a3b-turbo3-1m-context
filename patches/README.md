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

## append-to-slot.patch

Adds `append_to_slot` parameter to the `/completion` endpoint. When `true`, query tokens are appended directly to the slot's existing cached KV state, avoiding the need to resend and re-tokenize the full 905K base prompt.

Key changes:
- `server-task.h/.cpp`: Add `append_to_slot` bool to `task_params`
- `server-context.cpp`: Add `n_input_total` and `append_tokens` to `server_slot`. On append mode, combine cached + query tokens once in `SLOT_STATE_STARTED`, then use persistent buffer throughout the processing loop. Replace all `slot.task->n_tokens()` with `slot.n_input_total` (19 replacements)

This reduces TTFT from ~6.3s to ~3.8s by eliminating:
- Network transfer of 1.25M chars
- Server-side tokenization of 905K tokens (~3s)

The remaining ~3s is `prompt cache update` (server-internal prefix matching over 905K tokens).
