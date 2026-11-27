# Qwen OpenCode specialist on one existing H200

Public/synthetic commissioning configuration. Not a production or benchmark acceptance claim.

One existing H200, TP1, BF16, native confidential GPU mode, no debug/admin shell. A separate exact-route authenticated gateway exposes models, chat completions, and authenticated public qualification metadata. Model network egress is closed; the backend is not published. Credential verification uses only the SHA256 of a controller-generated high-entropy Qwen-only key. Real client credentials remain in the separate Perimeter, never OpenCode or this repository.

Exact public source modelpacks are dm-verity verified. Measured startup code performs the complete original rank32/alpha64 FP32 adapter merge into BF16, shard by shard, with input SHA256 checks, 256 target checks, independent FP64 delta samples, finite-value checks, saved-file reload equality and untouched-tensor equality. The arithmetic matches PEFT0.21.2 vanilla safe_merge's BF16 rounding sequence. This is a measured derivation into temporary enclave RAM, NOT a separately dm-verity-mounted merged checkpoint and NOT yet a qualified numerical-equivalence claim against a full unmerged PEFT model. No private Hugging Face upload/token is needed. No pickle, custom model code, quantization, target dropping or hosted fallback.

The exact base training revision is unknown because the adapter's revision field is null. This selected base revision is pinned for a functional pilot, not an exact authors' benchmark reproduction. Original source licenses and notices are preserved in the source modelpacks/merged output. PEFT reference: https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/layer.py .

Extension candidate limits: 262144 total context, 32768 output, two active generations, 4MiB JSON input. This is a bring-up profile, not a lifetime token/spend ceiling. Actual prompt tokens plus output reservation are validated by vLLM. The larger-context profile requires live qualification; the paired benchmark remains separate. Named forced tools intentionally return finish_reason=stop in vLLM0.34.0; automatic tool calls return tool_calls. Preserve length/error reasons and validate IDs/arguments instead of relabeling upstream semantics.

16 CPU/128GiB/six-disk platform layout uses base and adapter packages plus a task-only 1GB compatibility disk that remains locked/unformatted and absent from application mounts. Fresh CPU/workload and separate GPU appraisal are still required; topology arithmetic alone is not acceptance.

OpenCode2.2.7 runs in a distinct attested Bunker, with fresh configuration/state, no inherited credentials, disabled automatic updates and no external provider/MCP/plugin access. Any updates use STOP-confirmed replacement of this evaluation instance only. Never borrow or modify the existing document GPU.

Sources and tests are published for review. The release workflow measures/signs the configuration; its success does not prove runtime/model/tool correctness. Actual evidence belongs in the commissioning report.
