from __future__ import annotations

import secrets
import signal
from pathlib import Path
from urllib.parse import urlparse

from gradio import networking
from gradio.tunneling import CURRENT_TUNNELS


PORT = 7860
URL_FILE = Path(__file__).resolve().parent.parent / ".public_tunnel" / "url"


def main() -> None:
    url = networking.setup_tunnel(
        local_host="127.0.0.1",
        local_port=PORT,
        share_token=secrets.token_urlsafe(32),
        share_server_address=None,
        share_server_tls_certificate=None,
    )
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".gradio.live"):
        raise RuntimeError(f"Unexpected Gradio share URL: {url}")

    tunnel = CURRENT_TUNNELS[-1]
    if tunnel.proc is None:
        raise RuntimeError("Gradio tunnel process was not created")

    URL_FILE.write_text(f"{url}\n", encoding="utf-8")
    print(f"[public] {url}", flush=True)

    def stop(signum: int, _frame: object) -> None:
        tunnel.kill()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return_code = tunnel.proc.wait()
    URL_FILE.unlink(missing_ok=True)
    raise RuntimeError(f"Gradio tunnel exited with status {return_code}")


if __name__ == "__main__":
    main()
