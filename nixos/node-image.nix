# OurobourOS node image config (R2_BRINGUP.md §10 WP5).
#
# Stateless cattle: squashfs root, identity derived from hardware each
# boot, roles never persisted (Art. 1). getty autologin spawns
# `ouro-agent --stdio-tty` — a booted node with a login joins the graph.
{ lib, pkgs, ouro-agent, ... }:

let
  crimson = "DC143C";

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
    cat > /run/ouro/issue <<ISSUE

\e[31m        the machine that remakes itself.\e[0m

\e[31;1m  >> $line\e[0m

  node $node_id · measured admission · secret: $secret_state
  enroll: $enroll_state

ISSUE
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
  '';

  ouro-enroll = pkgs.writeShellScriptBin "ouro-enroll" ''
    #!/usr/bin/env bash
    # Consume the labeled OURO partition: HMAC secret + master pubkey.
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
    if [ -s "$mnt/authorized_keys" ]; then
      "${pkgs.coreutils}/bin/install" -d -m 700 -o ouro -g ouro /home/ouro/.ssh
      "${pkgs.coreutils}/bin/install" -m 600 -o ouro -g ouro \
        "$mnt/authorized_keys" /home/ouro/.ssh/authorized_keys
      status keys-installed
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
    exec ${ouro-agent}/bin/ouro-agent --stdio-tty
  '') // { shellPath = "/bin/ouro-shim"; };
in
{
  imports = [ ];

  # live-ISO shape (dd-able USB; R2_BRINGUP.md §10)
  image.fileName = lib.mkForce "ourobouros-node.iso";
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

  # sleep is not a node state (route B leftover absorbed here)
  systemd.targets.sleep.enable = false;
  systemd.targets.suspend.enable = false;
  systemd.targets.hibernate.enable = false;

  # the node user; its login shell is the agent shim
  users.groups.ouro = { };
  users.users.ouro = {
    isNormalUser = true;
    group = "ouro";
    description = "OurobourOS node";
    shell = ouro-shim;
    # WP7 debug image only: the flash-time key (OURO partition) is the
    # production path; this baked key lets `ssh` in for journal debugging.
    openssh.authorizedKeys.keys = [
      ''command="/run/current-system/sw/bin/bash" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILSQiBkmB9O4KP66DDXjcJtlNPguZZuDSY2vutp0zoJG ouro-wp7-debug-shell''
      "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILSQiBkmB9O4KP66DDXjcJtlNPguZZuDSY2vutp0zoJG ouro-wp7-test"
    ];
  };
  services.getty = {
    autologinUser = "ouro";
    # our generated issue (brand + measured state), not the default
    greetingLine = "";
    extraArgs = [ "-f" "/run/ouro/issue" ];
  };

  # raw-serial path (runbook §3 WP3: "SSH pty (or raw serial)"):
  # autologin on ttyS0 runs the same shim — a node with no monitor joins
  # over a serial line, brand banner included.
  systemd.services."serial-getty@ttyS0" = {
    enable = true;
    wantedBy = [ "getty.target" ];
    serviceConfig.ExecStart = lib.mkForce
      "${pkgs.util-linux}/bin/agetty -n --autologin ouro -f /run/ouro/issue --keep-baud 115200,57600,38400,9600 %I $TERM";
  };

  # serial console for headless boots + the QEMU prove-out
  boot.kernelParams = [
    "quiet"
    "console=tty0"
    "console=ttyS0,115200n8"
  ];
  # stamp the issue fresh at each boot after brand picks the tagline
  systemd.services.ouro-brand = {
    description = "OurobourOS boot brand: random tagline + console palette";
    before = [ "getty@tty1.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-palette}";
      ExecStartPost = "${ouro-brand}/bin/ouro-brand";
    };
  };

  systemd.services.ouro-probe = {
    description = "OurobourOS derived identity + measured sleep capability";
    before = [ "ouro-brand.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-probe}/bin/ouro-probe";
    };
  };

  systemd.services.ouro-enroll = {
    description = "OurobourOS enrollment from OURO-labeled partition";
    before = [ "ouro-brand.service" "sshd.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${ouro-enroll}/bin/ouro-enroll";
      StandardOutput = "console";
      StandardError = "console";
    };
  };

  # the wire: master reaches the node over ssh -T; keys only
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

  environment.systemPackages = [ ouro-agent ouro-brand ouro-probe ouro-enroll ouro-shim ];

  # login(1)-friendly: register the custom shell
  environment.etc."shells".text = lib.mkAfter ''
    /run/current-system/sw/bin/ouro-shim
  '';

  system.stateVersion = "25.05";
}
