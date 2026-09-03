#!/usr/bin/env bash
# OuroborOS node-image flasher (R2_BRINGUP.md §10 WP6).
#
# Usage: tools/flash.sh <image.iso> <device> [enroll-dir]
#   image.iso   nix build .#node-image result path (ISO, dd-able)
#   device      /dev/sdX — WILL BE OVERWRITTEN, no confirmations
#   enroll-dir  directory with `secret` (+ optional `authorized_keys`,
#               `head` = registry IP:PORT); default: ./enroll
#
# Writes the image, then appends an OURO-labeled FAT partition (if the
# image leaves room) carrying the HMAC secret + head SSH pubkey.
# Sticks are keys: whoever holds the stick can join the cluster.
set -euo pipefail

die() { echo "flash: $*" >&2; exit 1; }

[ "$#" -ge 2 ] || die "usage: $0 <image.iso> <device> [enroll-dir]"
IMG=$1
DEV=$2
ENROLL=${3:-./enroll}

[ -f "$IMG" ] || die "image not found: $IMG"
[ -b "$DEV" ] || die "not a block device: $DEV"
[ -f "$ENROLL/secret" ] || die "missing $ENROLL/secret (32B hex HMAC secret)"
[[ "$(tr -d ' \n\r' < "$ENROLL/secret")" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "$ENROLL/secret must be exactly 64 hex chars"

echo "flash: overwriting $DEV with $IMG in 3s — Ctrl+C to abort"
sleep 3

# 1. image
dd if="$IMG" of="$DEV" bs=4M status=progress conv=fsync

# 2. OURO partition (needs free space after the image). lsblk may report
# the pre-dd table's partitions until partprobe — parse defensively.
SIZE=$(lsblk -bno size "$DEV" | head -n1)
P1SIZE=$(lsblk -bno size "${DEV}1" 2>/dev/null | head -n1 || echo 0)
FREE=$((SIZE - P1SIZE))
OURO_WRITTEN=0
if [ "$FREE" -gt 33554432 ]; then
  sfdisk --append "$DEV" <<PART
label: dos
type=c
PART
  partprobe "$DEV" || true
  LAST=$(lsblk -rno NAME,TYPE "$DEV" | awk '$2=="part"{n=$1} END{print n}')
  mkfs.vfat -n OURO "/dev/$LAST"
  mnt=$(mktemp -d)
  mount "/dev/$LAST" "$mnt"
  install -m 600 "$ENROLL/secret" "$mnt/secret"
  [ -f "$ENROLL/authorized_keys" ] && install -m 644 "$ENROLL/authorized_keys" "$mnt/authorized_keys"
  # Bus join: `head` names the registry (IP:PORT) the tail registers with.
  [ -f "$ENROLL/head" ] && install -m 644 "$ENROLL/head" "$mnt/head"
  sync && umount "$mnt" && rmdir "$mnt"
  OURO_WRITTEN=1
  echo "flash: OURO partition written (/dev/$LAST)"
else
  echo "flash: WARNING no room for OURO partition — use a second USB labeled OURO (enroll dir: $ENROLL)"
fi

# 3. verify readback — sectors 1-8 only: sector 0 is the MBR, which
# sfdisk legitimately rewrites when it appends the OURO partition.
dd if="$DEV" bs=512 skip=1 count=8 2>/dev/null | cmp - <(dd if="$IMG" bs=512 skip=1 count=8 2>/dev/null) \
  || die "readback mismatch — the image on $DEV is not trustworthy, reflash"
if [ "$OURO_WRITTEN" = 1 ]; then
  echo "flash: $DEV ready with OURO enrollment. boot-order is the only step left."
else
  die "$DEV has the image but NO OURO partition — the tail will refuse the wire. Reflash or enroll via a second USB."
fi
