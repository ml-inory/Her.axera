# SenseVoice ASR Provider 接入说明

本文档说明如何在后端启用 `sensevoice` ASR Provider。模型与部署方式来源于 Hugging Face 仓库：<https://huggingface.co/AXERA-TECH/SenseVoice>。该仓库当前说明要求使用 Python 3.12，并安装 `pyaxengine==0.1.3rc2`；模型支持 AX650N 与 AX630C。

## 1. 后端接入方式

后端新增 ASR Provider：`sensevoice`。

调用链路：

```text
POST /v1/asr/transcriptions
  -> ASRService
  -> SenseVoiceProvider
  -> SENSEVOICE_REPO_PATH/python/main.py 或 SENSEVOICE_REPO_PATH/main.py
  -> 解析 stdout 中的识别文本
```

Provider 会将上传音频保存为临时文件，然后调用 SenseVoice 的 Python 入口：

```bash
python3 main.py --input /path/to/audio.wav --language auto
```

如果开启流式模型，会额外传入：

```bash
--streaming
```

## 2. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| ENABLE_SENSEVOICE_ASR | `false` | 是否注册 `sensevoice` Provider。 |
| DEFAULT_ASR_PROVIDER | `mock_asr` | 可设置为 `sensevoice`，让后端默认使用 SenseVoice。 |
| SENSEVOICE_REPO_PATH | 空 | SenseVoice 仓库本地路径。 |
| SENSEVOICE_PYTHON | `python3` | 运行 SenseVoice demo 的 Python 解释器，推荐指向 Python 3.12 虚拟环境。 |
| SENSEVOICE_LANGUAGE | `auto` | 默认语言，可选 `auto`、`zh`、`en`、`yue`、`ja`、`ko`。 |
| SENSEVOICE_TIMEOUT_SEC | `60` | 单次识别超时时间。 |
| SENSEVOICE_STREAMING | `false` | 是否传入 `--streaming` 并使用 streaming 模型。 |

## 3. 部署步骤

### 3.1 获取模型仓库

```bash
git clone https://huggingface.co/AXERA-TECH/SenseVoice /opt/models/SenseVoice
```

如果环境无法直接访问 Hugging Face，可在可联网机器下载后拷贝到部署机器。

### 3.2 安装 SenseVoice 运行依赖

进入 SenseVoice 仓库，根据 Hugging Face 页面中的说明安装依赖。典型方式如下：

```bash
cd /opt/models/SenseVoice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyaxengine==0.1.3rc2
```

根据 Hugging Face 页面最新说明，AXERA 部署需要安装 `pyaxengine==0.1.3rc2`。如果使用官方命令，推荐在 Python 3.12 虚拟环境中安装。

### 3.3 验证 SenseVoice demo

先独立验证 SenseVoice demo 可以运行：

```bash
cd /opt/models/SenseVoice/python
python3 main.py --input ../example/en.mp3 --language auto
```

若仓库入口位于根目录，也可以使用：

```bash
cd /opt/models/SenseVoice
python3 main.py --input example/en.mp3 --language auto
```

成功后再接入后端。

### 3.4 配置后端

```bash
export ENABLE_SENSEVOICE_ASR=true
export SENSEVOICE_REPO_PATH=/opt/models/SenseVoice
export SENSEVOICE_PYTHON=/opt/models/SenseVoice/.venv/bin/python
export SENSEVOICE_LANGUAGE=auto
# 如需默认使用 SenseVoice：
export DEFAULT_ASR_PROVIDER=sensevoice
```

启动后端：

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 4. API 调用示例

显式指定 SenseVoice：

```bash
curl -X POST "http://127.0.0.1:8080/v1/asr/transcriptions" \
  -F "audio=@input.wav" \
  -F "provider=sensevoice" \
  -F "language=auto" \
  -F "enable_timestamps=true"
```

如果 `DEFAULT_ASR_PROVIDER=sensevoice`，可省略 `provider` 字段。

查询 Provider：

```bash
curl "http://127.0.0.1:8080/v1/asr/providers"
```

## 5. 语言映射

后端会将常见语言代码映射到 SenseVoice 支持的语言参数：

| API 输入 | SenseVoice 参数 |
| --- | --- |
| `auto` | `auto` |
| `zh`、`zh-CN`、`zh-TW` | `zh` |
| `en`、`en-US`、`en-GB` | `en` |
| `yue` | `yue` |
| `ja`、`ja-JP` | `ja` |
| `ko`、`ko-KR` | `ko` |

## 6. 错误排查

| 错误码 | 常见原因 |
| --- | --- |
| `sensevoice_not_configured` | 未设置 `SENSEVOICE_REPO_PATH`。 |
| `sensevoice_path_not_found` | `SENSEVOICE_REPO_PATH` 指向的目录不存在。 |
| `sensevoice_repo_invalid` | 仓库中未找到 `python/main.py` 或 `main.py`。 |
| `sensevoice_invocation_failed` | `SENSEVOICE_PYTHON` 不存在或不可执行。 |
| `sensevoice_timeout` | 识别超过 `SENSEVOICE_TIMEOUT_SEC`。 |
| `sensevoice_provider_error` | SenseVoice demo 进程返回非 0 状态码。 |
| `sensevoice_empty_result` | demo 执行成功但后端未能从 stdout 解析出文本。 |

