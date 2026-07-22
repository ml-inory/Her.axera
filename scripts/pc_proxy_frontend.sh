#!/usr/bin/env bash
# Serve Her.axera frontend with built-in API proxy to AX board backend.
# All browser requests go to localhost:7860, avoiding proxy interference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_URL="${BACKEND_URL:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/pc_proxy_frontend.sh --backend-url http://AX650_IP:8000

Options:
  --backend-url URL   AX650 backend URL (required)
  --host HOST         Bind host. Default: 0.0.0.0
  --port PORT         Listen port. Default: 7860
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-url) shift; BACKEND_URL="${1%/}" ;;
    --host) shift; HOST="$1" ;;
    --port) shift; PORT="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "${BACKEND_URL}" ]] || { usage >&2; exit 2; }

exec "${PYTHON_BIN}" - "${BACKEND_URL}" "${HOST}" "${PORT}" "${REPO_ROOT}/frontend/static" <<'PYEOF'
import http.server
import urllib.request, urllib.error
import sys

BACKEND = sys.argv[1]
HOST, PORT = sys.argv[2], int(sys.argv[3])
STATIC_DIR = sys.argv[4]

# Bypass system proxy for backend calls
PROXY_HANDLER = urllib.request.ProxyHandler({})
OPENER = urllib.request.build_opener(PROXY_HANDLER)

print(f"[proxy] backend: {BACKEND}")
print(f"[proxy] listen:  http://{HOST}:{PORT}")
print(f"[open] http://127.0.0.1:{PORT}/")

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith(("/v1/", "/health", "/ui")):
            return self._proxy()
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/v1/"):
            return self._proxy()
        self.send_error(404)

    def do_OPTIONS(self):
        return self._proxy()

    def _proxy(self):
        url = BACKEND + self.path
        body = None
        clen = int(self.headers.get("Content-Length", 0))
        if clen > 0:
            body = self.rfile.read(clen)
        try:
            req = urllib.request.Request(url, data=body, method=self.command)
            for k, v in self.headers.items():
                if k.lower() not in ("host", "content-length"):
                    req.add_header(k, v)
            resp = OPENER.open(req, timeout=30)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, str(e))

    def log_message(self, fmt, *args):
        print(f"[{self.command}] {args[0]}")

httpd = http.server.HTTPServer((HOST, PORT), ProxyHandler)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n[proxy] stopped")
PYEOF
