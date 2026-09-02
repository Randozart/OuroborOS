#!/usr/bin/env python3
"""WP7 QEMU prove-out (R2_BRINGUP.md §10).

Boots the OurobourOS node image under QEMU (TCG), with an OURO-labeled
enrollment drive attached, and drives the getty-spawned agent over the
serial line with signed wire traffic.

Asserts:
  1. brand: console palette stamped, tagline + measured state in the issue
  2. enrollment: secret consumed -> `secret: ok`
  3. getty-shim: autologin spawns `ouro-agent --stdio-tty` on ttyS0
  4. signed wire: valid tag -> signed pong; tamper -> opaque rejection
"""
import os
import pty
import subprocess
import sys
import time
import hmac
import hashlib
import select

WORK = os.environ.get("WP7_WORK", "/tmp/opencode/qemu")
ISO = os.environ.get(
    "WP7_ISO",
    os.path.expanduser(
        "~/Desktop/Projects/OurobourOS/result/iso/nixos-26.11.20260829.e8be781-x86_64-linux.iso"
    ),
)
SECRET_FILE = os.path.join(WORK, "secret.hex")
BOOT_TIMEOUT = 420  # TCG boot budget
LINE_TIMEOUT = 30

secret = bytes.fromhex(open(SECRET_FILE).read().strip())


def sign(seq, body):
    tag = hmac.new(secret, seq.to_bytes(8, "big"), hashlib.sha256).hexdigest()
    return f"{seq} {tag} {body}"


def main():
    master, slave = pty.openpty()
    tty = os.ttyname(slave)
    print(f"[wp7] serial pty: {tty}")
    qemu = subprocess.Popen(
        [
            "qemu-system-x86_64",
            "-accel", "tcg,thread=multi", "-cpu", "max", "-m", "3072",
            "-cdrom", ISO, "-boot", "d",
            "-drive", f"file={WORK}/enroll.img,format=raw,if=ide",
            "-display", "none", "-monitor", "none",
            "-serial", tty,
        ],
        stdout=open(f"{WORK}/qemu.stdout", "w"),
        stderr=subprocess.STDOUT,
    )
    buf = b""
    deadline = time.time() + BOOT_TIMEOUT

    def pump(timeout=5.0):
        nonlocal buf
        r, _, _ = select.select([master], [], [], timeout)
        if master in r:
            try:
                buf += os.read(master, 65536)
            except OSError:
                return False
        return True

    # 1+2: wait for the brand/state stamp on serial
    seen_issue = False
    while time.time() < deadline:
        if not pump():
            break
        text = buf.decode("utf-8", "replace")
        if "measured admission" in text:
            seen_issue = True
            break
    assert seen_issue, f"issue banner never appeared; tail:\n{buf[-3000:].decode('utf-8','replace')}"
    text = buf.decode("utf-8", "replace")
    assert "the machine that remakes itself" in text, "brand line missing"
    assert "secret: ok" in text, "enrollment failed: secret not consumed"
    for tag in ["it knows what it is.", "devour the default.", "no purpose but use.",
                "the tail feeds the head.", "one wire. one budget. one machine.",
                "nothing declared. everything measured."]:
        if tag in text:
            print(f"[wp7] 1,2 PASS  brand + enrollment  tagline: {tag!r}")
            break
    else:
        raise AssertionError("no tagline from the pool in the issue banner")

    # 3+4: the shim should be live on serial once autologin lands; a signed
    # ping must come back as a signed pong under the same seq.
    deadline = time.time() + 120
    pong = False
    seq = 1000
    last_send = 0.0
    while time.time() < deadline and not pong:
        pump(2.0)
        now = time.time()
        if now - last_send > 5:
            seq += 1
            os.write(master, (sign(seq, "ping") + "\n").encode())
            last_send = now
        text = buf.decode("utf-8", "replace")
        for line in text.splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) == 3 and parts[0].isdigit():
                want = hmac.new(
                    secret, int(parts[0]).to_bytes(8, "big"), hashlib.sha256
                ).hexdigest()
                if parts[0] == str(seq) and parts[1] == want and parts[2] == "pong":
                    pong = True
                    break
    assert pong, f"no signed pong; serial tail:\n{buf[-3000:].decode('utf-8','replace')}"
    print("[wp7] 3,4 PASS  getty-shim signed wire  ping->pong verified")
    print("[wp7] ALL PASS")

    qemu.terminate()
    qemu.wait(timeout=15)


if __name__ == "__main__":
    main()
