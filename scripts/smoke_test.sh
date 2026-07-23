#!/usr/bin/env bash
# Her.axera End-to-End Smoke Test
# Verifies backend health, OpenAI-compatible endpoints, and WebSocket connectivity.
#
# Usage:
#   scripts/smoke_test.sh [--host HOST] [--port PORT]
#
# Exit codes:
#   0 - all tests passed
#   1 - one or more tests failed

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

_pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); }
_fail() { echo -e "  ${RED}FAIL${NC} $1 — $2"; FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); }

echo "============================================"
echo " Her.axera Smoke Test"
echo " Target: ${BASE}"
echo "============================================"
echo ""

# ── 1. Health Check ───────────────────────────────────────────────
echo "[1] Health Check"

if HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health" 2>/dev/null); then
    if [ "$HEALTH" = "200" ]; then
        _pass "/health returns 200"
    else
        _fail "/health" "got HTTP ${HEALTH}"
    fi
else
    _fail "/health" "connection refused — is the backend running?"
    echo ""
    echo "Make sure the backend is running: scripts/ax650_run_backend.sh"
    exit 1
fi

# ── 2. Provider Listing ───────────────────────────────────────────
echo "[2] Provider Listing"

MODELS=$(curl -s "${BASE}/v1/models" 2>/dev/null || echo '{}')
ASR_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len([p for p in d.get('providers',{}).get('asr',[]) if p.get('name','mock')!='mock_asr']))" 2>/dev/null || echo "0")
LLM_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len([p for p in d.get('providers',{}).get('llm',[]) if p.get('name','mock')!='mock_llm']))" 2>/dev/null || echo "0")
TTS_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len([p for p in d.get('providers',{}).get('tts',[]) if p.get('name','mock')!='mock_tts']))" 2>/dev/null || echo "0")

echo "  ASR providers (non-mock): ${ASR_COUNT}"
echo "  LLM providers (non-mock): ${LLM_COUNT}"
echo "  TTS providers (non-mock): ${TTS_COUNT}"

if [ "$ASR_COUNT" -gt 0 ] || [ "$LLM_COUNT" -gt 0 ] || [ "$TTS_COUNT" -gt 0 ]; then
    _pass "/v1/models returns providers"
else
    _fail "/v1/models" "no non-mock providers registered (use mock for basic test)"
fi

# ── 3. OpenAI-Compatible LLM Chat ─────────────────────────────────
echo "[3] OpenAI-Compatible LLM Chat"

LLM_RESP=$(curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mock_llm","messages":[{"role":"user","content":"你好"}],"max_tokens":20}' 2>/dev/null || echo '{}')

LLM_TEXT=$(echo "$LLM_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")

if [ -n "$LLM_TEXT" ]; then
    echo "  Reply: ${LLM_TEXT:0:60}..."
    _pass "/v1/chat/completions returns content"
else
    _fail "/v1/chat/completions" "empty response"
fi

# ── 4. OpenAI-Compatible TTS Speech ───────────────────────────────
echo "[4] OpenAI-Compatible TTS Speech"

HTTP_CODE=$(curl -s -o /tmp/her_smoke_tts.mp3 -w "%{http_code}" -X POST "${BASE}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"model":"mock_tts","input":"你好世界","voice":"alloy","response_format":"mp3"}' 2>/dev/null || echo "000")

TTS_SIZE=$(stat -c%s /tmp/her_smoke_tts.mp3 2>/dev/null || echo "0")
rm -f /tmp/her_smoke_tts.mp3

if [ "$HTTP_CODE" = "200" ] && [ "$TTS_SIZE" -gt 100 ]; then
    _pass "/v1/audio/speech returns audio (${TTS_SIZE} bytes)"
else
    _fail "/v1/audio/speech" "HTTP ${HTTP_CODE}, size ${TTS_SIZE}"
fi

# ── 5. WebSocket Connectivity ─────────────────────────────────────
echo "[5] WebSocket Connectivity"

WS_URL="ws://${HOST}:${PORT}/v1/dialogue/ws"
WS_CHECK=$(python3 -c "
import asyncio, websockets, sys
async def check():
    try:
        async with websockets.connect('${WS_URL}', ping_timeout=3) as ws:
            return 'ok'
    except Exception as e:
        return str(e)
print(asyncio.run(check()))
" 2>/dev/null || echo "websockets not installed")

if [ "$WS_CHECK" = "ok" ]; then
    _pass "WebSocket /v1/dialogue/ws connected"
else
    if echo "$WS_CHECK" | grep -q "websockets not installed"; then
        echo -e "  ${YELLOW}SKIP${NC} WebSocket (websockets package not installed)"
    else
        _fail "WebSocket" "${WS_CHECK:0:80}"
    fi
fi

# ── 6. Wake Word API ──────────────────────────────────────────────
echo "[6] Wake Word API"

WW_LIST=$(curl -s "${BASE}/v1/wakewords" 2>/dev/null || echo '{}')
WW_COUNT=$(echo "$WW_LIST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('wake_words',[])))" 2>/dev/null || echo "0")

if curl -s "${BASE}/v1/wakewords" > /dev/null 2>&1; then
    _pass "/v1/wakewords accessible (${WW_COUNT} registered)"
else
    _fail "/v1/wakewords" "endpoint unreachable"
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
