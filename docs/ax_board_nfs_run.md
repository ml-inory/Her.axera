# AX650 Backend + PC Frontend Runbook

This runbook keeps the source tree on the development machine and runs the Her.axera backend from an AX650 board through an NFS mount. The PC runs the static frontend and connects to the board backend.

## Development Machine

Export the repository parent directory:

```bash
sudo apt-get install -y nfs-kernel-server
sudo mkdir -p /srv/nfs/her-axera
sudo mount --bind /opt/rzyang/Github/Her.axera /srv/nfs/her-axera
echo "/srv/nfs/her-axera *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee /etc/exports.d/her-axera.exports
sudo exportfs -ra
```

Start the backend locally when you only need browser UI validation:

```bash
cd /opt/rzyang/Github/Her.axera/backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## AX650 Board

Mount the exported tree:

```bash
mkdir -p /mnt/her-axera
mount -t nfs -o nolock,tcp <DEV_MACHINE_IP>:/srv/nfs/her-axera /mnt/her-axera
cd /mnt/her-axera
```

Prepare backend dependencies:

```bash
scripts/ax650_setup_backend.sh
```

The base backend install includes `silero-vad-axera`. The setup script keeps the venv isolated by default, then links the board-provided `axengine` package into that venv when available so the default environment does not inherit system `torch`. The generated `backend/.env` also includes `/soc/lib` in `LD_LIBRARY_PATH` so AX650 can load `libax_engine.so` through `AxEngineExecutionProvider`.

Optionally download selected models. The downloader uses `huggingface_hub` and defaults to `https://hf-mirror.com`:

```bash
scripts/ax650_setup_backend.sh --models "sensevoice kokoro speaker"
```

Start the backend:

```bash
scripts/ax650_run_backend.sh --host 0.0.0.0 --port 8080
```

Validate from the board:

```bash
curl http://127.0.0.1:8080/health
```

Install a managed systemd service after setup:

```bash
scripts/ax650_install_service.sh --enable --start
```

Useful service commands:

```bash
systemctl status her-axera-backend.service
journalctl -u her-axera-backend.service -f
sudo systemctl restart her-axera-backend.service
```

## PC Frontend

Run the static frontend on the PC and point it at the AX650 backend:

```bash
cd /opt/rzyang/Github/Her.axera
scripts/pc_run_frontend.sh --backend-url http://<AX_BOARD_IP>:8080
```

Open the URL printed by the script. It includes an `?api=...` parameter so the browser stores the board backend URL automatically.

You can also open the backend-hosted UI directly:

```text
http://<AX_BOARD_IP>:8080/ui/
```

Use mock providers first to verify the WebSocket/UI path, then enable AXEngine-backed providers in `backend/.env`.

## Download AX Models On The Board

The setup script installs the model download dependency automatically. To install it manually on the AX board:

```bash
cd /mnt/her-axera
python3 -m pip install -r backend/requirements-model-download.txt
```

Download all AXERA model repositories. The downloader defaults to `https://hf-mirror.com` and writes a backend env snippet:

```bash
python3 backend/tools/download_ax_models.py all \
  --root /opt/models/her-axera \
  --env-file backend/.env.models
```

Download only selected models:

```bash
python3 backend/tools/download_ax_models.py sensevoice fireredasr kokoro zipvoice speaker \
  --env-file backend/.env.models
```

Preview files without downloading:

```bash
python3 backend/tools/download_ax_models.py speaker --dry-run
```

Apply the generated settings after reviewing them:

```bash
cat backend/.env.models >> backend/.env
```

The script uses `huggingface_hub.snapshot_download`, so interrupted downloads can resume. Override the mirror when needed:

```bash
HF_ENDPOINT=https://huggingface.co python3 backend/tools/download_ax_models.py speaker
```

Do not switch `DEFAULT_ASR_PROVIDER`, `DEFAULT_TTS_PROVIDER`, or `DEFAULT_SPEAKER_PROVIDER` to real providers until the corresponding runtime dependencies have also been installed and the provider-specific validation command passes.
