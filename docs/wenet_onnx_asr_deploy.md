# WeNet ONNX ASR Deployment Validation

This project can use the ONNX runner from `https://github.com/ml-inory/wenet.axera.git` as an optional ASR provider named `wenet_onnx`.

## 1. Prepare the WeNet AXera Repo

```bash
git clone https://github.com/ml-inory/wenet.axera.git /opt/wenet.axera
```

Install the base backend dependencies and the optional ONNX runtime dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-wenet-onnx.txt
```

## 2. Prepare ONNX Artifacts

The provider expects the same artifact layout used by `wenet.axera/run_ort.py`:

```text
onnx_model/
  encoder_offline.onnx
  encoder_online.onnx    # only needed when WENET_ONLINE=true
  decoder.onnx           # only needed for attention_rescoring
  config.yaml
units.txt
```

Use the export flow in `wenet.axera/export_onnx.py` to generate `onnx_model/` from a WeNet checkpoint.

Example using the upstream default AISHELL pretrained model:

```bash
cd /opt/wenet.axera
python export_onnx.py --output_onnx_dir /opt/wenet.axera/onnx_model
```

This creates:

```text
onnx_model/
  config.yaml
  decoder.onnx
  encoder_offline.onnx
  encoder_online.onnx
pretrained/aishell_u2pp_conformer_exp/
  units.txt
```

## 3. Validate ONNX Directly

Before starting this backend, validate the ONNX runner with the upstream script:

```bash
cd /opt/wenet.axera
python run_ort.py \
  --input demo.wav \
  --config /path/to/onnx_model/config.yaml \
  --vocab /path/to/units.txt \
  --onnx_dir /path/to/onnx_model \
  --mode ctc_prefix_beam_search
```

Expected behavior: the script prints `ASR Result: ...`.

## 4. Enable the Backend Provider

Set these environment variables for the backend:

```bash
export ENABLE_WENET_ASR=true
export DEFAULT_ASR_PROVIDER=wenet_onnx
export WENET_REPO_PATH=/opt/wenet.axera
export WENET_ONNX_DIR=/path/to/onnx_model
export WENET_CONFIG_PATH=/path/to/onnx_model/config.yaml
export WENET_VOCAB_PATH=/path/to/units.txt
export WENET_MODE=ctc_prefix_beam_search
export WENET_ONLINE=false
export WENET_ORT_PROVIDERS=CPUExecutionProvider
```

Start the backend:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Validate provider registration:

```bash
curl http://127.0.0.1:8080/v1/asr/providers
```

Validate transcription through this backend:

```bash
curl -X POST http://127.0.0.1:8080/v1/asr/transcriptions \
  -F "audio=@/opt/wenet.axera/demo.wav" \
  -F "provider=wenet_onnx" \
  -F "language=zh-CN" \
  -F "enable_timestamps=true"
```

## Notes

- `attention_rescoring` requires `decoder.onnx`.
- `WENET_ONLINE=true` requires `encoder_online.onnx`.
- `WENET_CALIB_DATA_PATH` can be set to collect calibration inputs using the upstream runner's calibration path.
- The backend dynamically imports `ort_common.py` from `WENET_REPO_PATH`; the external repository is not vendored into this project.
- If export fails with `TypeError: export() got an unexpected keyword argument 'dynamo'`, the local PyTorch version does not support that `torch.onnx.export` argument. Remove the three `dynamo=False` arguments in `wenet.axera/export_onnx.py` or use a PyTorch version whose export API accepts it.
