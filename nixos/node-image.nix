# OuroborOS node image config (R2_BRINGUP.md §10 WP5).
#
# Stateless cattle: squashfs root, identity derived from hardware each
# boot, roles never persisted (Art. 1). getty autologin spawns
# `ouro-agent --stdio-tty` — a booted node with a login joins the graph.
{ lib, pkgs, config, ouro-agent, ... }:

let
  crimson = "DC143C";

  # The serpent — the hand-retouched brand logo
  # (docs/brand/ascii-logo-ramp-80-retouched.txt), served from the
  # store so the boot screen shows the machine's face. 40 lines on an
  # 80x25 console scrolls: the serpent plays as the intro and the
  # final frame is tagline + measured state (that's the point).
  ouroLogo = ../docs/brand/ascii-logo-ramp-80-retouched.txt;

  # Remap the 16-color console palette slot 1 (red) to crimson so the
  # bare TTY gets true crimson with plain `\e[31m` escapes; SSH
  # sessions get real truecolor.
  ouro-palette = pkgs.writeShellScript "ouro-palette" ''
    printf '\\e]P1%s' '${crimson}' > /dev/console 2>/dev/null || true
  '';

  ouro-brand = pkgs.writeShellScriptBin "ouro-brand" ''
    #!/usr/bin/env bash
    # Pick the boot's tagline at random (kernel entropy; stateless-safe),
    # expose it to the agent + banner, stamp the issue file.
    set -euo pipefail
    mkdir -p /run/ouro
    pool=/etc/ouro/taglines
    line=$("${pkgs.coreutils}/bin/shuf" -n1 "$pool" 2>/dev/null \
      || head -n1 "$pool")
    printf '%s' "$line" > /run/ouro/tagline
    node_id="$(cat /run/ouro/node_id 2>/dev/null || echo unknown)"
    secret_state=REFUSED
    [ -s /run/ouro/secret ] && secret_state=ok
    enroll_state="$(cat /run/ouro/enroll-status 2>/dev/null || echo no-enroll-run)"
    nics="$(cat /run/ouro/nics 2>/dev/null || echo unknown)"
    # The right/below text column: OUROBOROS wordmark + backronym +
    # taglines + measured state, built PLAIN (no ESC) so width math is
    # honest. Segment colors are applied at print time by row index.
    textcol=$(mktemp)
    {
      printf '%s\n' \
        '   ▄▄▄▄                                        ▄▄▄▄      ▄▄▄▄▄' \
        ' ▄█▀▀████▄                 █▄                ▄█▀▀████▄  ██▀▀▀▀█▄' \
        ' ██    ██       ▄          ██          ▄     ██    ██   ▀██▄  ▄▀' \
        ' ██    ██ ██ ██ ████▄▄███▄ ████▄ ▄███▄ ████▄ ██    ██     ▀██▄▄' \
        ' ██    ██ ██ ██ ██   ██ ██ ██ ██ ██ ██ ██    ██    ██   ▄   ▀██▄' \
        '  ▀████▀ ▄▀██▀█▄█▀  ▄▀███▀▄████▀▄▀███▀▄█▀     ▀████▀    ▀██████▀'
      printf '\n'
      printf ' OUROBOROS: One Unified Runtime Orchestrating\n'
      printf '            a Bunch Of Random Old Servers\n'
      printf '\n'
      printf '        the machine that remakes itself.\n'
      printf '\n'
      printf '  >> %s\n' "$line"
      printf '\n'
      printf '  node %s · measured admission · secret: %s\n' "$node_id" "$secret_state"
      printf '  enroll: %s\n' "$enroll_state"
      printf '  nics: %s\n' "$nics"
    } > "$textcol"
    mapfile -t tcol < "$textcol"
    ntxt=''${#tcol[@]}
    # segment color by row index (0-based): wordmark rows bold crimson,
    # taglines crimson, the rest plain. Emits only the OPEN escape;
    # close is always $'\e[0m'.
    tseg() {
      case "$1" in
        0|1|2|3|4|5) printf '\e[1;31m' ;;
        10)          printf '\e[31m' ;;
        12)          printf '\e[31;1m' ;;
        *)           : ;;
      esac
    }
    tpaint() { # row, line -> painted line (color if the row has one)
      local seg
      seg=$(tseg "$1")
      if [ -n "$seg" ]; then
        printf '%s%s\e[0m\n' "$seg" "$2"
      else
        printf '%s\n' "$2"
      fi
    }

    # console width decides the layout: >=140 cols -> side-by-side
    # (logo left, text right); narrower -> stacked (serpent, then text).
    if cols=$(stty size < /dev/console 2>/dev/null | awk '{print $2}'); then
      :; else cols=0; fi
    nlogo=$(wc -l < ${ouroLogo})
    {
      printf '\n'
      if [ "''${cols:-0}" -ge 140 ] 2>/dev/null; then
        top=$(( (nlogo - ntxt) / 2 ))
        r=1
        while read -r lrow; do
          # pad by CHAR count — ramp glyphs are 3-byte UTF-8, %-80s
          # would pad to bytes and the text column would go ragged
          pad=$(( 80 - ''${#lrow} ))
          [ "$pad" -lt 0 ] && pad=0
          logo_cell=$(printf '%s%*s' "$lrow" "$pad")
          ti=$(( r - 1 - top ))
          if [ "$ti" -ge 0 ] && [ "$ti" -lt "$ntxt" ]; then
            printf '\e[31m%s\e[0m  %s\n' "$logo_cell" "$(tpaint "$ti" "''${tcol[$ti]}")"
          else
            printf '\e[31m%s\e[0m\n' "$logo_cell"
          fi
          r=$(( r + 1 ))
        done < ${ouroLogo}
      else
        # stacked: serpent, then the text column (wordmark reintroduced)
        printf '\e[31m'
        cat ${ouroLogo}
        printf '\e[0m\n'
        for ti in "''${!tcol[@]}"; do
          tpaint "$ti" "''${tcol[$ti]}"
        done
      fi
      printf '\n'
    } > /run/ouro/issue
  '';

  ouro-probe = pkgs.writeShellScriptBin "ouro-probe" ''
    #!/usr/bin/env bash
    # Derived identity (never stored): node_id = sha256(SMBIOS uuid | MAC).
    # Roles are not recorded here — they are priced at schedule time.
    set -euo pipefail
    mkdir -p /run/ouro
    uuid="$(cat /sys/class/dmi/id/product_uuid 2>/dev/null || true)"
    nic="$(ls /sys/class/net | grep -v '^lo$' | head -n1 || true)"
    mac="$(cat "/sys/class/net/$nic/address" 2>/dev/null || true)"
    printf '%s|%s' "$uuid" "$mac" \
      | "${pkgs.coreutils}/bin/sha256sum" | cut -c1-16 > /run/ouro/node_id
    # measured sleep capability: WoL support + suspend blacklist
    wol=false
    nic_real="$nic"
    [ -e "/sys/class/net/$nic/device" ] && \
      "${pkgs.ethtool}/bin/ethtool" "$nic_real" 2>/dev/null | grep -q 'Supports Wake-on: .*p' && wol=true
    printf '%s' "$wol" > /run/ouro/wol
    # measured NIC census: name + carrier per interface (bus-join drill)
    nics=""
    for n in /sys/class/net/*; do
      [ -e "$n" ] || continue
      name="$(basename "$n")"
      c="$(cat "$n/carrier" 2>/dev/null || echo '?')"
      nics="$nics $name=$c"
    done
    printf '%s' "''${nics# }" > /run/ouro/nics
  '';

  ouro-enroll = pkgs.writeShellScriptBin "ouro-enroll" ''
    #!/usr/bin/env bash
    # Consume the labeled OURO partition: HMAC secret + head pubkey.
    # Missing partition => no secret => agent refuses the wire (WP2 gate).
    set -euo pipefail
    status() {
      printf '%s' "$1" > /run/ouro/enroll-status 2>/dev/null || true
      echo "ouro-enroll: $1" > /dev/console 2>/dev/null || true
    }
    mkdir -p /run/ouro
    # udev race: by-label links may lag at early boot — wait for them
    status searching
    dev=""
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      dev="$("${pkgs.util-linux}/bin/findfs" 'LABEL=OURO' 2>/dev/null || true)"
      [ -n "$dev" ] && break
      sleep 1
    done
    if [ -z "$dev" ]; then
      status "no-partition (agent will refuse the wire)"
      exit 0
    fi
    status "found $dev"
    mnt=/run/ouro/enroll
    mkdir -p "$mnt"
    if ! "${pkgs.util-linux}/bin/mount" -o ro "$dev" "$mnt" 2>/run/ouro/mount.err; then
      status "mount-failed: $(cat /run/ouro/mount.err 2>/dev/null | tail -c 120)"
      exit 1
    fi
    status mounted
    if [ -s "$mnt/secret" ]; then
      "${pkgs.coreutils}/bin/install" -m 600 -o ouro -g ouro \
        "$mnt/secret" /run/ouro/secret
      status secret-installed
    else
      status "no-secret-file-on-partition"
    fi
    # authorized_keys (+ optional debug_authorized_keys) — built fresh
    # every boot so the file is idempotent and the optional debug key
    # (opt-in: only if the stick carries enroll/debug_authorized_keys)
    # never accumulates across reboots. Wire key first, debug after.
    if [ -s "$mnt/authorized_keys" ] || [ -s "$mnt/debug_authorized_keys" ]; then
      "${pkgs.coreutils}/bin/install" -d -m 700 -o ouro -g ouro /home/ouro/.ssh
      : > /home/ouro/.ssh/authorized_keys
      [ -s "$mnt/authorized_keys" ] && \
        cat "$mnt/authorized_keys" >> /home/ouro/.ssh/authorized_keys
      [ -s "$mnt/debug_authorized_keys" ] && \
        cat "$mnt/debug_authorized_keys" >> /home/ouro/.ssh/authorized_keys
      "${pkgs.coreutils}/bin/chmod" 600 /home/ouro/.ssh/authorized_keys
      "${pkgs.coreutils}/bin/chown" ouro:ouro /home/ouro/.ssh/authorized_keys
      status keys-installed
    fi
    if [ -s "$mnt/head" ]; then
      "${pkgs.coreutils}/bin/install" -m 600 -o ouro -g ouro \
        "$mnt/head" /run/ouro/head
      status head-installed
    else
      status "no-head-file"
    fi
    "${pkgs.util-linux}/bin/umount" "$mnt" || true
    status complete
  '';

  # The getty "shell" for the autologin user IS the agent: login = join.
  # types.shellPackage check = isDerivation && hasAttr "shellPath".
  ouro-shim = (pkgs.writeShellScriptBin "ouro-shim" ''
    #!/usr/bin/env bash
    export OURO_SECRET_FILE=/run/ouro/secret
    export OURO_TAGLINE="$(cat /run/ouro/tagline 2>/dev/null || true)"
    # Bus join: the enroll partition's `head` file names the registry.
    if [ -s /run/ouro/head ]; then
      exec ${ouro-agent}/bin/ouro-agent --stdio-tty \
        --head "$(cat /run/ouro/head)"
    else
      exec ${ouro-agent}/bin/ouro-agent --stdio-tty
    fi
  '') // { shellPath = "/bin/ouro-shim"; };
in
{
  imports = [ ];

  # live-ISO shape (dd-able USB; R2_BRINGUP.md §10)
  image.fileName = lib.mkForce "ouroboros-node.iso";
  isoImage = {
    makeEfiBootable = true;
    makeUsbBootable = true;
  };
  fileSystems."/" = {
    device = "tmpfs";
    fsType = "tmpfs";
    options = [ "mode=0755" ];
  };

  # boot
  boot.loader.timeout = lib.mkForce 1;

  # stateless cattle: nothing to mutate
  nix.enable = false;

  # Networking: any ethernet NIC, DHCP, don't block boot on it. The bus
  # link (head_link) retries every 5s until the address lands.
  # QEMU/TCG slirp is deterministic — MAC 52:54:00:* is always 10.0.2.15
  # with the host at 10.0.2.2 — so QEMU NICs get a static lease applied
  # by a scripted oneshot (networkd races login under TCG and the prove
  # window is honest about it). Real hardware keeps DHCP.
  networking.useNetworkd = true;
  systemd.network.enable = true;
  # The stock NixOS firewall (default ON, ssh-only) blocked the head's
  # task channel to the agent port (FIRST_LIGHT.md blocker 9). Tails
  # are appliances among owned machines: the signed wire is the
  # gatekeeper (Art. 10) and the fabric is LAN-isolated
  # (DMA_ROADMAP.md). No host firewall.
  networking.firewall.enable = false;
  systemd.network.networks."80-ouro" = {
    matchConfig.Type = "ether";
    networkConfig.DHCP = "yes";
    linkConfig.RequiredForOnline = "no";
  };
  systemd.services.ouro-net = {
    description = "OuroborOS static slirp address (QEMU MACs)";
    wantedBy = [ "multi-user.target" ];
    before = [ "multi-user.target" ];
    path = [ pkgs.iproute2 ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      say() { echo "ouro-net: $1" > /dev/console 2>/dev/null || true; }
      # virtio_net may load after this unit runs — wait for a QEMU MAC
      # (slirp is deterministic: 10.0.2.15/24, host at 10.0.2.2).
      say "waiting for a QEMU-MAC NIC"
      for i in $(seq 1 45); do
        applied=0
        for nic in /sys/class/net/*; do
          n="$(basename "$nic")"
          [ "$n" = "lo" ] && continue
          mac="$(cat "$nic/address" 2>/dev/null || true)"
          say "saw $n mac=$mac"
          case "$mac" in
            52:54:00:*)
              ip link set "$n" up && say "raised $n"
              ip address add 10.0.2.15/24 dev "$n" 2>/dev/null || say "addr add failed"
              ip route add default via 10.0.2.2 dev "$n" 2>/dev/null || say "route add failed"
              applied=1
              ;;
          esac
        done
        [ "$applied" = 1 ] && { say "slirp address applied"; exit 0; }
        sleep 2
      done
      say "no QEMU-MAC NIC appeared in 90s"
    '';
  };

  # sleep is not a node state (route B leftover absorbed here)
  systemd.targets.sleep.enable = false;
  systemd.targets.suspend.enable = false;
  systemd.targets.hibernate.enable = false;

  # the node user; its login shell is the agent shim
  users.groups.ouro = { };
  users.users.ouro = {
    isNormalUser = true;
    group = "ouro";
    extraGroups = [ "video" "render" ];
    description = "OuroborOS node";
    shell = ouro-shim;
    # No baked keys — SSH access arrives via the OURO partition's
    # authorized_keys at enroll time. Whoever holds the stick holds
    # the node (R2_BRINGUP.md §8).
  };

  # GPU compute: Intel NEO runtime (OpenCL 3.0 + Level Zero) for
  # tails with Intel iGPUs. The agent detects GPUs via detect_gpus()
  # and reports them over the bus (GPU_CLAIM.md, WP-G2).
  hardware.graphics.extraPackages = [ pkgs.intel-compute-runtime ];

  # OpenCL loader discovery: the ocl-icd loader defaults to
  # /etc/OpenCL/vendors, but NixOS installs ICD registrations under
  # /run/opengl-driver/etc/OpenCL/vendors. The agent runs from the
  # getty shim — no login profile — so the variable must live in the
  # system environment (found live: Platform::list error 10, zero ICDs).
  environment.variables.OCL_ICD_VENDORS = "/run/opengl-driver/etc/OpenCL/vendors";

  # NVIDIA: the HP Pavilion carries a GTX 1060 6GB (Pascal) — the
  # driver makes nvidia-smi (and the OpenCL ICD) exist, which is all
  # detect_gpus() and the agent's OpenCL path need (GPU_CLAIM.md
  # WP-N2). Proprietary module: Pascal predates the open kernel
  # module (Turing+). Headless compute — no modesetting, no X runs;
  # nomodeset above stays (NVIDIA compute is KMS-free). No
  # finegrained RTD3 on Pascal — the 1060 idles on standard PCI PM.
  nixpkgs.config.allowUnfree = true; # the driver is unfreeRedistributable
  services.xserver.videoDrivers = [ "nvidia" ]; # triggers the module; no X
  hardware.nvidia = {
    open = false;
    modesetting.enable = false;
    nvidiaSettings = false;
    package = config.boot.kernelPackages.nvidiaPackages.legacy_580;
  };
  services.getty = {
    autologinUser = "ouro";
    # NO -f issue here: the agent's isatty banner is the only banner
    # (agetty's issue print collided with it — FIRST_LIGHT.md blocker
    # 10). greetingLine stays empty so nothing prints before the shim.
    greetingLine = "";
  };

  # Shelf tails live lid-closed: closing the lid must park the wyrm,
  # not suspend it (a laptop tail is an appliance with a hinge).
  services.logind.lidSwitch = "ignore";
  services.logind.lidSwitchExternalPower = "ignore";

  # raw-serial path (runbook §3 WP3: "SSH pty (or raw serial)"):
  # autologin on ttyS0 runs the same shim — a node with no monitor joins
  # over a serial line, brand banner included.
  systemd.services."serial-getty@ttyS0" = {
    enable = true;
    wantedBy = [ "getty.target" ];
    serviceConfig.ExecStart = lib.mkForce
      "${pkgs.util-linux}/bin/agetty -n --autologin ouro -f /run/ouro/issue --keep-baud 115200,57600,38400,9600 %I $TERM";
  };

  # serial console for headless boots + the QEMU prove-out.
  # nomodeset: tails are headless compute — KMS buys nothing and panics
  # odd iGPUs into boot loops (first real tail, an HP Pavilion, looped
  # GRUB→panic on real hardware 2026-09-03). Plain VGA text console.
  boot.kernelParams = [
    "quiet"
    "console=tty0"
    "console=ttyS0,115200n8"
    "nomodeset"
  ];
  # stamp the issue fresh at each boot after brand picks the tagline
  systemd.services.ouro-brand = {
    description = "OuroborOS boot brand: random tagline + console palette";
    before = [ "getty@tty1.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-palette}";
      ExecStartPost = "${ouro-brand}/bin/ouro-brand";
    };
  };

  systemd.services.ouro-probe = {
    description = "OuroborOS derived identity + measured sleep capability";
    before = [ "ouro-brand.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-probe}/bin/ouro-probe";
    };
  };

  systemd.services.ouro-enroll = {
    description = "OuroborOS enrollment from OURO-labeled partition";
    before = [ "ouro-brand.service" "sshd.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-enroll}/bin/ouro-enroll";
      StandardOutput = "console";
      StandardError = "console";
    };
  };

  # the wire: head reaches the node over ssh -T; keys only
  services.openssh = {
    enable = true;
    settings = {
      PasswordAuthentication = false;
      PermitRootLogin = "no";
    };
  };

  # console + branding payload
  environment.etc."ouro/taglines".text = ''
    it knows what it is.
    devour the default.
    no purpose but use.
    the tail feeds the head.
    one wire. one budget. one machine.
    nothing declared. everything measured.
  '';
  console = {
    earlySetup = true;
    colors = with builtins; [
      "000000" "${crimson}" "555555" "aaaaaa"
      "8b0000" "dc143c" "ff6b81" "ffffff"
      "000000" "dc143c" "555555" "aaaaaa"
      "8b0000" "dc143c" "ff6b81" "ffffff"
    ];
  };

  # ocl-icd: the OpenCL loader the agent dlopens at runtime; the
  # intel-compute-runtime ICD above registers the actual driver
  environment.systemPackages = [ ouro-agent ouro-brand ouro-probe ouro-enroll ouro-shim pkgs.iproute2 pkgs.ocl-icd ];

  # login(1)-friendly: register the custom shell
  environment.etc."shells".text = lib.mkAfter ''
    /run/current-system/sw/bin/ouro-shim
  '';

  system.stateVersion = "25.05";
}
