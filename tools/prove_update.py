#!/usr/bin/env python3
"""WP-U6 QEMU prove: the tail remakes itself (UPDATE_ROADMAP §tests).

Two scenarios, both against a REAL booted image with a REAL host registry:

  A. GUARD — cdrom boot + an 8MB OURO drive: an image push must be
     REFUSED before any bytes are written (no partition table on the
     anchor drive; the ISO would overrun it). Refusal must come back
     as `err update`.

  B. SELF-REFLASH — cdrom boot + a 2GB spare OURO drive whose FAT sits
     BEYOND the ISO's end: the tail receives the ISO over frames,
     stages it on the anchor, passes the start-sector guard, writes
     [0, iso_size) onto that drive, readback-verifies, reboots, and
     rejoins the bus with the same identity (find_by_ip).

The cdrom shape means the rewritten drive is not the boot medium —
the literal boot-medium rewrite is U7's hardware receipt; everything
up to it (verify → frames → sha → guard → raw write → readback →
reboot → rejoin) is proven here.

Rootless. Host-side needs: qemu-system-x86_64, mkfs.vfat, mtools.
Usage: U6_ISO=/path/to/iso python3 tools/prove_update.py
"""

import hashlib
import hmac
import json
import os
import pty
import select
import socket
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("U6_WORK", "/tmp/opencode/u6")
ISO = os.environ.get("U6_ISO")
UPDATE = os.path.join(REPO, "tools", "ouro-update")
SECRET_FILE = os.path.join(REPO, "enroll", "secret")
PUBKEY = os.path.join(REPO, "keys", "update.signing.pub")
REG_BIN = os.path.join(REPO, "target", "release", "ouro-registry")
BOOT_TIMEOUT = 420


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{cmd[0]} failed: {r.stderr[-400:]}")
    return r.stdout


def free_port():
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def secret():
    return bytes.fromhex(open(SECRET_FILE).read().strip())


def signed_ping(host, port, timeout=20):
    sec = secret()
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(timeout)
    try:
        body = b"ping"
        tag = hmac.new(sec, (1).to_bytes(8, "big") + body, hashlib.sha256).hexdigest()
        s.sendall(f"1 {tag} ping\n".encode())
        return s.recv(4096).decode().strip().endswith("pong")
    finally:
        s.close()


def build_fat_image(path, size_mb, registry_port):
    """OURO-labeled FAT with the enroll files, rootless (mtools)."""
    with open(path, "wb") as f:
        f.seek(size_mb * 1024 * 1024 - 1)
        f.write(b"\0")
    sh("mkfs.vfat", "-n", "OURO", path)
    subprocess.run(["mcopy", "-i", path, "-o", SECRET_FILE, "::secret"], check=True)
    subprocess.run(["mcopy", "-i", path, "-o", PUBKEY, "::authorized_keys"], check=True)
    head = os.path.join(WORK, "head")
    with open(head, "w") as f:
        f.write(f"10.0.2.2:{registry_port}\n")
    subprocess.run(["mcopy", "-i", path, "-o", head, "::head"], check=True)


def build_spare_drive(path, registry_port, iso_size, fat_start=None):
    """2GB drive, MBR with one FAT partition. Default: the FAT starts
    BEYOND the ISO's end (guard must pass; the raw write must stop
    before the FAT). With an explicit fat_start BELOW the ISO's end the
    guard must REFUSE — scenario A proves exactly that rail, after a
    full successful staging (no ENOSPC shortcut). Rootless: MBR entry
    written by hand, FAT built with mtools then dd'd in at the offset."""
    sector = 512
    if fat_start is None:
        fat_start = iso_size // sector + 4096
    total = 2 * 1024 * 1024 * 1024
    fat_len = total - fat_start * sector
    fat_img = path + ".fat"
    build_fat_image(fat_img, fat_len // (1024 * 1024), registry_port)

    with open(path, "wb") as f:
        f.seek(total - 1)
        f.write(b"\0")
    mbr = bytearray(512)
    # one partition entry: FAT32 LBA, starting past the ISO's end
    mbr[446:462] = bytes([0x00, 0xFE, 0xFF, 0xFF, 0x0C, 0xFE, 0xFF, 0xFF])
    mbr[454:462] = struct.pack("<II", fat_start, fat_len // sector)
    mbr[510:512] = b"\x55\xAA"
    with open(path, "r+b") as f:
        f.write(mbr)
        f.seek(fat_start * sector)
        f.write(open(fat_img, "rb").read())
    os.remove(fat_img)
    return fat_start


def spawn_registry(port, state_path):
    return subprocess.Popen(
        [REG_BIN, "--addr", f"0.0.0.0:{port}", "--state", state_path],
        env={**os.environ, "OURO_SECRET_FILE": SECRET_FILE},
        stdout=open(os.path.join(WORK, "registry.log"), "a"),
        stderr=subprocess.STDOUT,
    )


def wait_agent(port, timeout=BOOT_TIMEOUT, what="agent", serial=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if signed_ping("127.0.0.1", port, timeout=3):
                return
        except OSError:
            pass
        if serial is not None:
            serial.pump(2.0)
        else:
            time.sleep(3)
    tail = serial.text()[-1800:] if serial is not None else "(no serial attached)"
    raise AssertionError(f"{what} task channel never came up on {port}\nserial tail:\n{tail}")


def push_image(port, iso_path):
    return subprocess.run(
        ["python3", UPDATE, "push", "image", "--node", "127.0.0.1",
         "--port", str(port), "--artifact", iso_path],
        capture_output=True, text=True, timeout=1800,
    )


class Serial:
    """pty attached to the VM's serial console; a passive observer.
    qemu gets the SLAVE path (a real terminal), we read the MASTER —
    ttyname(master) is /dev/ptmx (the multiplexer), not a terminal
    (found live: the VM's boot output went into the void and the agent
    seemed never to come up)."""

    def __init__(self, tag):
        self.master, self.slave = pty.openpty()
        self.fd = self.master
        self.tty = os.ttyname(self.slave)
        self.tag = tag
        self.buf = b""

    def pump(self, timeout=2.0):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if self.fd in r:
            try:
                self.buf += os.read(self.fd, 65536)
            except OSError:
                pass

    def text(self):
        return self.buf.decode("utf-8", "replace")


def boot_vmu(tag, serial, drives, fwd_port):
    """One cdrom boot + arbitrary extra drives, task channel forwarded."""
    cmd = ["qemu-system-x86_64", "-accel", "tcg,thread=multi", "-cpu", "max",
           "-m", "3072", "-cdrom", ISO, "-boot", "d",
           "-display", "none", "-monitor", "none", "-serial", serial.tty,
           "-device", "virtio-net-pci,netdev=n0",
           "-netdev", f"user,id=n0,hostfwd=tcp:127.0.0.1:{fwd_port}-:9500"]
    for d in drives:
        cmd += ["-drive", f"file={d},format=raw,if=virtio"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=False)


def main():
    os.makedirs(WORK, exist_ok=True)
    if not ISO:
        sys.exit("set U6_ISO=/path/to/ouroboros-node.iso")
    reg_port = free_port()
    state_path = os.path.join(WORK, "registry-state.json")
    registry = spawn_registry(reg_port, state_path)
    try:
        # ---------- Scenario A: the guard ----------
        # 2GB drive, FAT at 600MB — BELOW the ISO's end: the full
        # staging transfer must succeed, then the start-sector guard
        # must refuse the raw write. The anchor is never touched.
        print("[u6] A: the guard refuses to eat the anchor", flush=True)
        guard_drive = os.path.join(WORK, "guard.img")
        iso_size = os.path.getsize(ISO)
        build_spare_drive(guard_drive, reg_port, iso_size,
                          fat_start=(600 * 1024 * 1024) // 512)
        serial = Serial("A")
        fwd = free_port()
        qemu = boot_vmu("A", serial, [guard_drive], fwd)
        try:
            wait_agent(fwd, what="A agent", serial=serial)
            r = push_image(fwd, ISO)
            out = r.stdout + r.stderr
            for _ in range(6):
                serial.pump(2.0)
            refusal = ("err update" in out or "err update" in serial.text())
            overrun = "overrun" in out or "overrun" in serial.text()
            assert r.returncode != 0 and refusal, (
                f"the guard MUST refuse — rc={r.returncode}\n"
                f"tool: {out[-600:]}\nserial: {serial.text()[-800:]}")
            print(f"[u6] A PASS  guard refused the write (overrun detected: {overrun})",
                  flush=True)
        finally:
            qemu.terminate()
            try:
                qemu.wait(timeout=15)
            except subprocess.TimeoutExpired:
                qemu.kill()

        # ---------- Scenario B: self-reflash ----------
        print("[u6] B: the tail rewrites a disk it is running beside", flush=True)
        iso_size = os.path.getsize(ISO)
        spare = os.path.join(WORK, "spare.img")
        fat_start = build_spare_drive(spare, reg_port, iso_size)
        print(f"[u6] B setup: ISO {iso_size // (1024*1024)}MiB, FAT at "
              f"{fat_start * 512 // (1024*1024)}MiB on the spare", flush=True)
        serial = Serial("B")
        fwd = free_port()
        qemu = boot_vmu("B", serial, [spare], fwd)
        try:
            wait_agent(fwd, what="B agent", serial=serial)
            t0 = time.time()
            r = push_image(fwd, ISO)
            out = r.stdout + r.stderr
            assert "receipt" in out and '"status": "Success"' in out, (
                f"self-reflash receipt not Success: {out[-600:]}")
            print(f"[u6] B1 PASS  staged + guard passed + written + readback "
                  f"verified ({time.time()-t0:.0f}s)", flush=True)
            # the tail rebooted: the VM returns on the same cdrom ISO and
            # rejoins with the same identity (find_by_ip on 10.0.2.15).
            deadline = time.time() + 600
            rejoined = False
            while time.time() < deadline and not rejoined:
                try:
                    state = json.load(open(state_path))
                    for n in state.get("nodes", {}).values():
                        if n.get("last_seen", 0) > time.time() - 90:
                            rejoined = True
                except (OSError, json.JSONDecodeError):
                    pass
                if not rejoined:
                    time.sleep(5)
            assert rejoined, f"tail never rejoined; serial:\n{serial.text()[-1500:]}"
            print("[u6] B2 PASS  rebooted and rejoined the bus", flush=True)
        finally:
            qemu.terminate()
            try:
                qemu.wait(timeout=15)
            except subprocess.TimeoutExpired:
                qemu.kill()
        print("[u6] ALL PASS")
    finally:
        registry.terminate()
        try:
            registry.wait(timeout=10)
        except subprocess.TimeoutExpired:
            registry.kill()


if __name__ == "__main__":
    main()
