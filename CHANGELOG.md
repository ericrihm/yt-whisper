# Changelog

## 0.2.0 — 2026-10-19

### Added

- Native Apple-silicon CPU/int8 support without a failed CUDA initialization first.
- Optional `metal` extra using upstream MLX Whisper, with `auto`, `cpu`, `cuda` and `metal` selection in the CLI, API and TUI.
- Persisted backend policy for history re-runs and a scrollable transcription form.
- Documented native local-recording transcription with the upstream `mlx_whisper` CLI; no new course downloader, account import or recording service.

### Fixed

- Restrict NVIDIA dependencies to supported Linux/Windows architectures rather than requiring unavailable CUDA wheels on macOS.
- Constrain PyAV to the compatible16.x series after real audio decoding exposed the removed `metadata_errors` argument in PyAV19.0.1.
- Propagate explicit backend failures instead of silently switching execution targets.
- Serialize Metal model loading and decoding within the process because MLX Whisper's model cache is shared across threaded TUI runs.

### Limits

- Metal returns completed segments as a batch; in-flight decode is not cooperatively cancelled.
- The authored tiny.en sample qualifies functionality, not long-recording accuracy, timing or throughput. CUDA execution on other hosts was not part of this macOS qualification.
