# FireRedASR-AED ASR Provider 接入说明

本文档说明如何在后端启用 `fireredasr_aed` ASR Provider。模型和部署方式来源于 Hugging Face 仓库 `AXERA-TECH/FireRedASR-AED`，下载时优先使用 hf-mirror 镜像：<https://hf-mirror.com/AXERA-TECH/FireRedASR-AED>。

## 1. Provider 概览

后端新增 ASR Provider：`fireredasr_aed`。

调用链路：

```text
POST /v1/audio/transcriptions
  -> ASRService
  -> FireRedASRAEDProvider
  -> FIREREDASR_REPO_PATH/fireredasr_axmodel.py
  -> axmodel/encoder.axmodel + decoder_loop.axmodel
```

仓库说明中的关键约束：

- 运行平台：`AX650N`。
- Python：建议 `3.12`。
- 支持语言：中文、英文。
- 单段最长输入：`10s`；更长音频会由仓库内 VAD 逻辑切分后推理。
- 独立验证入口：`python test_ax_model.py`，识别结果写入 `hypo_axmodel.txt`。

## 2. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| ENABLE_FIREREDASR_ASR | `false` | 是否注册 `fireredasr_aed` Provider。 |
| DEFAULT_ASR_PROVIDER | `mock_asr` | 可设置为 `fireredasr_aed`，让后端默认使用 FireRedASR-AED。 |
| FIREREDASR_REPO_PATH | 空 | `AXERA-TECH/FireRedASR-AED` 仓库本地路径。 |
| FIREREDASR_MODEL_DIR | `<repo>/axmodel` | 模型文件目录。 |
| FIREREDASR_BEAM_SIZE | `1` | 解码 beam size。 |
| FIREREDASR_NBEST | `1` | 解码 nbest。 |
| FIREREDASR_DECODE_MAX_LEN | `128` | 最大解码 token 长度。 |
| FIREREDASR_MAX_AUDIO_SEC | `10` | 单段音频最大秒数。 |

`FIREREDASR_MODEL_DIR` 中必须包含：

- `encoder.axmodel`
- `decoder_loop.axmodel`
- `cmvn.ark`
- `dict.txt`
- `train_bpe1000.model`
- `pe.npy`

## 3. 获取模型仓库

使用 hf-mirror 克隆仓库：

```bash
git lfs install
git clone https://hf-mirror.com/AXERA-TECH/FireRedASR-AED /opt/models/FireRedASR-AED
```

如果需要用 Hugging Face CLI 下载，也要指定镜像 endpoint：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download AXERA-TECH/FireRedASR-AED \
  --local-dir /opt/models/FireRedASR-AED \
  --local-dir-use-symlinks False
```

下载后检查 `.axmodel` 是否为真实模型文件，而不是 Git LFS 指针：

```bash
ls -lh /opt/models/FireRedASR-AED/axmodel/*.axmodel
head -n 1 /opt/models/FireRedASR-AED/axmodel/encoder.axmodel
```

如果 `head` 输出类似 `version https://git-lfs.github.com/spec/v1`，说明 LFS 文件未拉取，需要在仓库目录执行：

```bash
cd /opt/models/FireRedASR-AED
git lfs pull
```

## 4. 安装依赖

推荐单独创建 Python 3.12 环境：

```bash
conda create -n fireredasr python=3.12 -y
conda activate fireredasr
sudo apt install libsndfile1
cd /opt/models/FireRedASR-AED
pip install -r requirements.txt
```

也可以使用后端提供的可选依赖清单：

```bash
cd /path/to/Her.axera/backend
pip install -r requirements-fireredasr-aed.txt
```

`axengine` wheel 来源于 AXERA-TECH 的 `pyaxengine` 发布包。如果部署环境无法访问 GitHub，可提前下载该 wheel 后离线安装，或复用板端已有的 `axengine` 包。

## 5. 独立验证

先验证 FireRedASR-AED 原始 demo 可以运行：

```bash
cd /opt/models/FireRedASR-AED
python test_ax_model.py
cat hypo_axmodel.txt
```

如需指定音频列表：

```bash
printf '%s\n' /path/to/input.wav > /tmp/firered_wavlist.txt
python test_ax_model.py --wavlist /tmp/firered_wavlist.txt
```

## 6. 后端启用

```bash
export ENABLE_FIREREDASR_ASR=true
export DEFAULT_ASR_PROVIDER=fireredasr_aed
export FIREREDASR_REPO_PATH=/opt/models/FireRedASR-AED
export FIREREDASR_MODEL_DIR=/opt/models/FireRedASR-AED/axmodel

cd /path/to/Her.axera/backend
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Provider 会在第一次请求时懒加载模型，启动服务时不会立即初始化 `axengine`。

## 7. OpenAI-Compatible 调用

显式指定 FireRedASR-AED：

```bash
curl -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \
  -F "file=@input.wav" \
  -F "model=fireredasr_aed" \
  -F "language=zh" \
  -F "response_format=verbose_json"
```

也可以用模型名触发自动路由：

```bash
curl -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \
  -F "file=@input.wav" \
  -F "model=fireredasr-aed-ax650n" \
  -F "response_format=json"
```

如果 `DEFAULT_ASR_PROVIDER=fireredasr_aed`，也可以继续传 OpenAI 默认的 `model=whisper-1`，由默认 Provider 执行。

## 8. 错误排查

| 错误码 | 常见原因 |
| --- | --- |
| `fireredasr_not_configured` | 未设置 `FIREREDASR_REPO_PATH`。 |
| `fireredasr_path_not_found` | `FIREREDASR_REPO_PATH` 指向的目录不存在。 |
| `fireredasr_model_invalid` | `axmodel` 目录缺少必要模型文件，或 LFS 文件未拉取。 |
| `fireredasr_dependency_missing` | 缺少 `torch`、`torchaudio`、`axengine`、`silero_vad_axera` 等运行依赖。 |
| `fireredasr_runner_init_failed` | `axengine` 初始化模型失败，常见于非 AX650N 环境或模型文件不完整。 |
| `fireredasr_transcription_failed` | 推理过程失败，需查看后端日志中的底层异常。 |
| `fireredasr_empty_result` | 推理完成但返回空文本。 |
