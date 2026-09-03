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
grep -Eq '^[0-9a-fA-F]{64}$' "$(tr -d ' \n' < "$ENROLL/secret"; echo)" \
  || die "$ENROLL/secret must be exactly 64 hex chars"

echo "flash: overwriting $DEV with $IMG in 3s — Ctrl+C to abort"
sleep 3

# 1. image
dd if="$IMG" of="$DEV" bs=4M status=progress conv=fsync

# 2. OURO partition (best effort: needs free space after the image)
SIZE=$(lsblk -bno size "$DEV")
PART_START_SECT=""
IFS=: read -r _ START _ _ < <(sfdisk -d "$DEV" 2>/dev/null | head -n1) || true

if FREE=$((SIZE - $(lsblk -bno size "${DEV}1" 2>/dev/null || echo SIZE))); [ "$FREE" -gt 33554432 ]; then
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
  echo "flash: OURO partition written (/dev/$LAST)"
else
  echo "flash: WARNING no room for OURO partition — use a second USB labeled OURO (enroll dir: $ENROLL)"
fi

# 3. verify readback
head -c 512 "$DEV" | cmp - <(head -c 512 "$IMG") && echo "flash: $DEV ready. boot-order is the only step left."
