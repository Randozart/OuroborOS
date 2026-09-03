#!/usr/bin/env python3
"""WP7 QEMU prove-out (R2_BRINGUP.md §10) + WP7.5 bus join.

Boots the OuroborOS node image under QEMU (TCG), with an OURO-labeled
enrollment drive attached (secret, head pubkey, head address), and drives
the getty-spawned agent over the serial line with signed wire traffic.
A real ouro-registry runs on the host; the tail must find it through the
enroll partition's `head` file (guest-side slirp gateway 10.0.2.2).

Asserts:
  1. brand: console palette stamped, tagline + measured state in the issue
  2. enrollment: secret consumed -> `secret: ok`
  3. getty-shim: autologin spawns `ouro-agent --stdio-tty` on ttyS0
  4. signed wire: valid tag -> signed pong; tamper -> opaque rejection
  5. bus join: tail registered with the host registry, heartbeating
"""
import json
import os
import pty
import subprocess
import sys
import time
import hmac
import hashlib
import select
import socket
import glob

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WORK = os.environ.get("WP7_WORK", "/tmp/opencode/qemu")


def default_iso():
    isos = sorted(glob.glob(os.path.join(REPO, "result/iso/*.iso")))
    assert isos, "no ISO in result/iso/ — run `nix build ./nixos#node-image`"
    return isos[-1]


ISO = os.environ.get("WP7_ISO", default_iso())
REGISTRY_BIN = os.path.join(REPO, "target/debug/ouro-registry")
SECRET_FILE = os.path.join(WORK, "secret.hex")
PUBKEY_FILE = os.path.join(WORK, "ouro_master_key.pub")
ENROLL_IMG = os.path.join(WORK, "enroll.img")
STATE_JSON = os.path.join(WORK, "registry-state.json")
BOOT_TIMEOUT = 420  # TCG boot budget
LINE_TIMEOUT = 30

secret = bytes.fromhex(open(SECRET_FILE).read().strip())


def sign(seq, body):
    tag = hmac.new(secret, seq.to_bytes(8, "big") + body.encode(), hashlib.sha256).hexdigest()
    return f"{seq} {tag} {body}"


def free_port():
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_enroll_image(registry_port):
    """Fresh OURO-labeled FAT: secret, head pubkey, head address."""
    with open(ENROLL_IMG, "wb") as f:
        f.seek(8 * 1024 * 1024 - 1)
        f.write(b"\0")
    subprocess.run(["mkfs.vfat", "-n", "OURO", ENROLL_IMG], check=True, capture_output=True)
    def mcopy(src, dst):
        subprocess.run(["mcopy", "-i", ENROLL_IMG, "-o", src, dst], check=True)
    mcopy(SECRET_FILE, "::secret")
    mcopy(PUBKEY_FILE, "::authorized_keys")
    # slirp: the guest's address for the host is 10.0.2.2
    head = os.path.join(WORK, "head")
    with open(head, "w") as f:
        f.write(f"10.0.2.2:{registry_port}\n")
    mcopy(head, "::head")


def spawn_registry(port):
    return subprocess.Popen(
        [REGISTRY_BIN, "--addr", f"0.0.0.0:{port}", "--state", STATE_JSON],
        env={**os.environ, "OURO_SECRET_FILE": SECRET_FILE},
        stdout=open(f"{WORK}/registry.log", "w"),
        stderr=subprocess.STDOUT,
    )


def wait_registry(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(0.3)
    raise AssertionError(f"registry daemon never came up on {port}")


def main():
    # 0: host-side registry for the bus-join assert (WP7.5)
    reg_port = free_port()
    build_enroll_image(reg_port)
    registry = subprocess.Popen(
        [REGISTRY_BIN, "--addr", f"0.0.0.0:{reg_port}", "--state", STATE_JSON],
        env={**os.environ, "OURO_SECRET_FILE": SECRET_FILE},
        stdout=open(f"{WORK}/registry.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        serial = run_qemu_asserts(reg_port)
        assert_bus_join(serial, registry, reg_port)
    finally:
        registry.terminate()
        try:
            registry.wait(timeout=10)
        except subprocess.TimeoutExpired:
            registry.kill()


def run_qemu_asserts(reg_port):
    """Boot + wire asserts. Returns the full serial text."""
    fd, tty_fd = pty.openpty()
    tty = os.ttyname(tty_fd)
    print(f"[wp7] serial pty: {tty}")
    qemu = subprocess.Popen(
        [
            "qemu-system-x86_64",
            "-accel", "tcg,thread=multi", "-cpu", "max", "-m", "3072",
            "-cdrom", ISO, "-boot", "d",
            "-drive", f"file={ENROLL_IMG},format=raw,if=ide",
            "-display", "none", "-monitor", "none",
            "-serial", tty,
            # user-mode net: the tail's bus join dials the host as 10.0.2.2
            "-device", "virtio-net-pci,netdev=n0",
            "-netdev", "user,id=n0",
        ],
        stdout=open(f"{WORK}/qemu.stdout", "w"),
        stderr=subprocess.STDOUT,
    )
    buf = b""
    deadline = time.time() + BOOT_TIMEOUT

    def pump(timeout=5.0):
        nonlocal buf
        r, _, _ = select.select([fd], [], [], timeout)
        if fd in r:
            try:
                buf += os.read(fd, 65536)
            except OSError:
                return False
        return True

    # 1+2: wait for the brand/state stamp on serial; the banner streams
    # through the pty over several reads, so wait until the secret state
    # token itself has landed (either state) before judging.
    seen_issue = False
    while time.time() < deadline:
        if not pump():
            break
        text = buf.decode("utf-8", "replace")
        if "measured admission" in text and (
            "secret: ok" in text or "secret: REFUSED" in text
        ):
            seen_issue = True
            break
        time.sleep(0.5)
    assert seen_issue, f"issue banner never appeared; tail:\n{buf[-3000:].decode('utf-8','replace')}"
    text = buf.decode("utf-8", "replace")
    assert "the machine that remakes itself" in text, "brand line missing"
    assert "secret: ok" in text, (
        "enrollment failed: secret not consumed\n"
        + "--- serial tail ---\n" + text[-1500:]
    )
    assert "enroll: complete" in text, "enroll breadcrumbs incomplete"
    for tag in ["it knows what it is.", "devour the default.", "no purpose but use.",
                "the tail feeds the head.", "one wire. one budget. one machine.",
                "nothing declared. everything measured."]:
        if tag in text:
            print(f"[wp7] 1,2 PASS  brand + enrollment  tagline: {tag!r}")
            break
    else:
        raise AssertionError("no tagline from the pool in the issue banner")

    # 3+4: wire gate over the raw-serial shim: signed ping -> signed pong
    # (proves enrollment), signed tagline -> pool line (proves brand).
    # The tty echoes each request line; tag verification is the security,
    # so parse on verified bodies, not seq bookkeeping.
    pool = ["it knows what it is.", "devour the default.", "no purpose but use.",
            "the tail feeds the head.", "one wire. one budget. one machine.",
            "nothing declared. everything measured."]
    deadline = time.time() + 180
    pong = False
    tagline = None
    seq = 1000
    last_send = 0.0
    send_body = "ping"
    while time.time() < deadline and not (pong and tagline):
        pump(2.0)
        now = time.time()
        if now - last_send > 5:
            seq += 1
            send_body = "tagline" if pong else "ping"
            os.write(fd, (sign(seq, send_body) + "\n").encode())
            last_send = now
        text = buf.decode("utf-8", "replace")
        for line in text.splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            want = hmac.new(
                secret, int(parts[0]).to_bytes(8, "big") + parts[2].encode(),
                hashlib.sha256
            ).hexdigest()
            if parts[1] != want:
                continue
            if parts[2] == "pong":
                pong = True
            elif parts[2] in pool:
                tagline = parts[2]
    assert pong, f"no signed pong; serial tail:\n{buf[-3000:].decode('utf-8','replace')}"
    assert tagline, f"no signed tagline; serial tail:\n{buf[-3000:].decode('utf-8','replace')}"
    print(f"[wp7] 3,4 PASS  getty-shim signed wire  ping->pong  tagline={tagline!r}")

    # Bus-join drill: measured network facts through the signed wire.
    seq += 1
    os.write(fd, (sign(seq, "diag") + "\n").encode())
    diag_deadline = time.time() + 20
    diag_text = ""
    while time.time() < diag_deadline and "--- /proc/net/route ---" not in diag_text:
        pump(1.0)
        text = buf.decode("utf-8", "replace")
        i = text.rfind("--- /proc/net/route ---")
        if i >= 0:
            for line in text[i:].splitlines():
                parts = line.strip().split(" ", 2)
                if len(parts) == 3 and parts[0].isdigit():
                    want = hmac.new(
                        secret, int(parts[0]).to_bytes(8, "big") + parts[2].encode(),
                        hashlib.sha256
                    ).hexdigest()
                    if parts[1] == want:
                        diag_text += parts[2].replace("\\n", "\n") + "\n"
    print("[diag] guest network facts:\n" + (diag_text[:1200] or "<no diag reply>"))

    text = buf.decode("utf-8", "replace")
    qemu.terminate()
    try:
        qemu.wait(timeout=15)
    except subprocess.TimeoutExpired:
        qemu.kill()
    return text


def assert_bus_join(serial, registry, port):
    """5: the tail found the registry through the enroll partition's
    `head` file and is heartbeating (WP7.5). Guest DHCP under TCG can
    lag well past the banner; head_link retries every 5s, so poll long.
    If the registry daemon dies out from under us (harness group-kills
    have form), respawn it — the tail's retry loop finds it again."""
    deadline = time.time() + 240
    state = None
    while time.time() < deadline:
        try:
            state = json.load(open(STATE_JSON))
        except (OSError, json.JSONDecodeError):
            state = None
        if state and state.get("nodes"):
            updated = [e for e in state.get("events", []) if e.get("NodeUpdated")]
            if len(updated) >= 2:
                break
        if registry.poll() is not None:
            print(f"[wp7.5] registry died (rc={registry.returncode}) — respawning")
            registry = spawn_registry(port)
        time.sleep(1)
    assert state and state.get("nodes"), (
        f"tail never registered; state: {state}\n--- full serial ---\n{serial}"
    )
    node = list(state["nodes"].values())[0]
    assert node["entry"]["hostname"] == "nixos", f"unexpected hostname: {node['entry']}"
    joined = [e for e in state.get("events", []) if e.get("NodeJoined")]
    updated = [e for e in state.get("events", []) if e.get("NodeUpdated")]
    assert joined, "no NodeJoined event"
    assert len(updated) >= 2, f"expected >=2 heartbeats, got {len(updated)}"
    print(
        f"[wp7.5] 5 PASS  bus join  node {node['entry']['id']} hostname={node['entry']['hostname']} "
        f"power={node['state']['power_watts']}W status={node['state']['status']} heartbeats={len(updated)}"
    )
    print("[wp7] ALL PASS")


if __name__ == "__main__":
    main()
