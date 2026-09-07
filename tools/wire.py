#!/usr/bin/env python3
"""Signed-wire client for OuroborOS tails (port 9500).

One signed line out, one signed line back — the same HMAC-SHA256 wire
the agent speaks (cluster/src/transport/auth.rs). Ping, diag, and task
dispatch without SSH.

Usage:
  wire.py ping [HOST]
  wire.py diag [HOST]
  wire.py task NAME [PAYLOAD] [HOST]
  wire.py raw BODY [HOST]

HOST defaults to the first heartbeat the registry has seen (reads
registry.json, freshest last_seen). Secret comes from OURO_SECRET_FILE
or ./enroll/secret.
"""

import hashlib
import hmac
import json
import os
import socket
import sys
import time

DEFAULT_PORT = 9500
TIMEOUT_SECS = 120


def load_secret() -> bytes:
    path = os.environ.get("OURO_SECRET_FILE", "enroll/secret")
    with open(path) as f:
        secret_hex = f.read().strip()
    return bytes.fromhex(secret_hex)


def newest_host() -> str:
    with open("registry.json") as f:
        state = json.load(f)
    nodes = state.get("nodes", {})
    if not nodes:
        sys.exit("wire: registry.json has no nodes")
    newest = max(nodes.values(), key=lambda r: r.get("last_seen", 0))
    ip = newest.get("entry", {}).get("ip", "?")
    nid = newest.get("entry", {}).get("id", "?")
    print(f"# target: {nid} @ {ip} (freshest heartbeat)", file=sys.stderr)
    return ip


def send(host: str, body: str, port: int = DEFAULT_PORT, timeout: int = TIMEOUT_SECS) -> str:
    secret = load_secret()
    seq = int(time.time()) & 0xFFFFFFFF
    tag = hmac.new(secret, seq.to_bytes(8, "big") + body.encode(), hashlib.sha256).hexdigest()
    line = f"{seq} {tag} {body}\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    resp_line = b""
    try:
        sock.connect((host, port))
        sock.sendall(line.encode())
        # One response line per request. The agent keeps the connection
        # open for pipelining, so read exactly ONE line and close —
        # waiting for close would block every call until the socket
        # timeout (found live: ping took 10s, tasks 120s).
        f = sock.makefile("rb")
        resp_line = f.readline()
    except socket.timeout:
        sock.close()
        sys.exit(f"wire: no reply from {host}:{port} within {timeout}s")
    finally:
        sock.close()

    resp = resp_line.decode().strip()
    if not resp:
        sys.exit("wire: 0 bytes — connection closed without a reply (agent task died?)")
    parts = resp.split(" ", 2)
    if len(parts) < 3:
        sys.exit(f"wire: malformed reply: {resp[:120]}")
    return parts[2]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "ping":
        host = rest[0] if rest else newest_host()
        print(send(host, "ping", timeout=10))
    elif cmd == "diag":
        host = rest[0] if rest else newest_host()
        print(send(host, "diag", timeout=30))
    elif cmd == "task":
        if not rest:
            sys.exit("wire: task needs a NAME")
        name = rest[0]
        payload = rest[1] if len(rest) > 1 else ""
        host = rest[2] if len(rest) > 2 else newest_host()
        task = json.dumps(
            {
                "id": f"wire-{int(time.time())}",
                "name": name,
                "payload": payload,
                "estimated_watts": 35,
                "estimated_seconds": 60,
            }
        )
        print(send(host, task))
    elif cmd == "raw":
        if not rest:
            sys.exit("wire: raw needs a BODY")
        host = rest[1] if len(rest) > 1 else newest_host()
        print(send(host, rest[0]))
    else:
        sys.exit(f"wire: unknown command {cmd!r} — use ping|diag|task|raw")


if __name__ == "__main__":
    main()
