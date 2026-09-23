ID: ISSUE-LOCAL-01M326DPAP4N14ZYWT1G7XEY01
Title: Pinned b10451 oracle refuses the on-disk APEX-I-Nano checkpoint (missing blk.64.ssm_conv1d.weight)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

oracle-dump at the pinned llama.cpp b10451 fails to load /mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf: 'llama_model_load: error loading model: missing tensor blk.64.ssm_conv1d.weight'. The file declares qwen35.nextn_predict_layers=1 with a PARTIAL MTP head, which the pin's loader refuses. The historical APEX token-exact gate therefore ran against a file revision (or an export variant, e.g. --no-mtp) that no longer matches the checkpoint on disk — the re-quantization-under-unchanged-name hazard. Consequence: there is NO oracle denominator for APEX-I-Nano at any prompt length today, and the 'APEX passes e2e' contrast used in the GSQ parity investigation (ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT evidence 5) is unverifiable. Fix direction: re-export/obtain an APEX variant the pin loads (e.g. --no-mtp), or record the sha256 of the exact gate artifact in docs/USAGE.md and restore it; then re-run the APEX token gate at both chunk-aligned (128) and boundary-crossing (63/65) prompt lengths. Discovered 2026-09-21: /tmp/apex_p0_bench.log, oracle stderr 'llama_model_load: error loading model: missing tensor blk.64.ssm_conv1d.weight'.

## Resolution

-
