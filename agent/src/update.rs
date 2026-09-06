//! Update intake — WP-U4 of WP-UPDATE ("the tail rewrites its own
//! genesis").
//!
//! Two artifacts, one contract:
//! - `agent`: 8MB hot-swap. Received over frames, verified, installed
//!   on the OURO partition, then `execv`'d in place — the login wire
//!   survives, the 9500 listener does not (CLOEXEC; clients reconnect).
//! - `image`: full ISO. Received to a staging file on the OURO
//!   partition, verified, written onto the tail's own boot medium
//!   (RAM-resident root; the ISO region is unmounted at runtime),
//!   readback-verified, then a clean reboot.
//!
//! Trust: the manifest's ed25519 signature is checked against the
//! pubkey baked into the image BEFORE any bytes are accepted; artifact
//! sha256 is checked against the manifest AFTER receipt. Trust follows
//! the signature, not the channel.
//!
//! Rollback: the baked-in ISO binary is the permanent fallback slot.
//! A pushed `agent-live` gets a boot counter on the OURO partition —
//! three boots without a successful bus registration and it is
//! discarded. No extra partitions, A/B semantics anyway.
use anyhow::{bail, Context, Result};
use ed25519_dalek::VerifyingKey;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ouro_cluster::transport::frames::{pump_recv, FrameSession, DEFAULT_WINDOW};
use ouro_cluster::transport::auth::Secret;
use ouro_cluster::update::{
    check_artifact_file, sha256_file, verify_signature, Artifact, Manifest,
};

/// Baked into the image at build time.
pub const UPDATE_PUBKEY_PATH: &str = "/etc/ouro/update.pub";
/// The persistent anchor: enroll mounts this at OURO_MOUNT (uid=ouro).
pub const OURO_PARTITION: &str = "/dev/disk/by-label/OURO";
pub const OURO_MOUNT: &str = "/mnt/ouro";
pub const RUN_DIR: &str = "/run/ouro";
/// Boot counter: three strikes and the live agent is discarded.
pub const BOOT_COUNTER_LIMIT: u64 = 3;

/// Path overrides for tests and QEMU proves — the defaults are the
/// image truths; a dev harness sets the env and gets a sandbox.
pub fn pubkey_path() -> PathBuf {
    std::env::var("OURO_UPDATE_PUBKEY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(UPDATE_PUBKEY_PATH))
}

pub fn ouro_mount() -> PathBuf {
    std::env::var("OURO_MOUNT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(OURO_MOUNT))
}

pub fn run_dir() -> PathBuf {
    std::env::var("OURO_RUN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(RUN_DIR))
}

/// Load the image-baked verification key.
pub fn load_pubkey() -> Result<VerifyingKey> {
    let path = pubkey_path();
    let hex = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let hex = hex.trim();
    if hex.len() != 64 {
        bail!("public key must be 64 hex chars, got {}", hex.len());
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).context("pubkey hex")?;
    }
    VerifyingKey::from_bytes(&bytes).context("update pubkey structure")
}

/// Verify a wire manifest against the image-baked pubkey and the
/// expected artifact kind. Signature first, always.
pub fn verify_manifest(wire: &str, expect: Artifact) -> Result<Manifest> {
    let pubkey = load_pubkey()?;
    let manifest = verify_signature(wire, &pubkey)?;
    let kind = Artifact::parse(&manifest.artifact)?;
    if kind != expect {
        bail!("manifest artifact {} != expected {}", kind.as_str(), expect.as_str());
    }
    if manifest.min_agent_protocol > ouro_cluster::update::PROTOCOL {
        bail!(
            "manifest needs protocol {} > ours {}",
            manifest.min_agent_protocol,
            ouro_cluster::update::PROTOCOL
        );
    }
    Ok(manifest)
}

/// OURO-partition paths for the live-agent slots.
pub fn live_paths() -> (PathBuf, PathBuf, PathBuf) {
    let mount = ouro_mount();
    (
        mount.join("agent-live"),
        mount.join("agent-update.json"),
        mount.join("agent-boot-count"),
    )
}

/// Boot decision as a pure function (the side-effecting wrapper is
/// [`boot_check`]).
#[derive(Debug, PartialEq)]
pub enum BootAction {
    /// Exec the live agent — it verified.
    ExecLive,
    /// Discard the live agent (counter exhausted or verification
    /// failed) and fall back to the baked-in binary.
    DiscardLive,
    /// No live agent present; run the baked-in binary.
    UseBakedIn,
}

pub fn boot_decision(live_exists: bool, counter: u64, verified: bool) -> BootAction {
    if !live_exists {
        return BootAction::UseBakedIn;
    }
    if counter >= BOOT_COUNTER_LIMIT {
        return BootAction::DiscardLive;
    }
    if verified {
        BootAction::ExecLive
    } else {
        BootAction::DiscardLive
    }
}

/// Startup handoff: if a pushed agent is resident and trustworthy,
/// return its path for exec; otherwise clean up any corpse so the
/// baked-in binary takes over. The boot counter is incremented here
/// and reset by [`reset_boot_counter`] once the bus accepts us. The
/// handoff never re-execs ourselves: when the current binary IS the
/// live agent, the check passes through (otherwise a hot-swap would
/// exec itself forever at every boot).
pub fn boot_check() -> Option<PathBuf> {
    let (live, manifest_path, counter_path) = live_paths();
    if !live.exists() {
        return None;
    }
    if let Ok(current) = std::fs::read_link("/proc/self/exe") {
        if let (Ok(live_canon), Ok(current_canon)) = (live.canonicalize(), current.canonicalize())
        {
            if live_canon == current_canon {
                return None; // we ARE the live agent
            }
        }
    }
    let counter = std::fs::read_to_string(&counter_path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
        + 1;
    let _ = std::fs::write(&counter_path, counter.to_string());

    let verified = (|| -> Result<bool> {
        let wire = std::fs::read_to_string(&manifest_path)?;
        let manifest = verify_manifest(&wire, Artifact::Agent)?;
        let (size, sha) = sha256_file(&live)?;
        let bytes_len_ok = manifest.size == size;
        Ok(bytes_len_ok && manifest.sha256 == sha)
    })()
    .unwrap_or(false);

    match boot_decision(true, counter, verified) {
        BootAction::ExecLive => Some(live),
        BootAction::DiscardLive => {
            let _ = std::fs::remove_file(&live);
            let _ = std::fs::remove_file(&manifest_path);
            let _ = std::fs::remove_file(&counter_path);
            None
        }
        BootAction::UseBakedIn => None,
    }
}

/// The tail registered with the head: the live agent has proven
/// itself. Zero the counter.
pub fn reset_boot_counter() {
    let (_, _, counter_path) = live_paths();
    let _ = std::fs::write(&counter_path, "0");
}
/// Stream a frame-mode payload into `staging`, then verify size +
/// sha256 against the manifest. Returns bytes received. The session is
/// borrowed so the caller can write the line-mode receipt on the same
/// socket afterwards (the final ack was the last frame-mode byte).
pub fn receive_artifact(
    session: &mut FrameSession<std::net::TcpStream>,
    manifest: &Manifest,
    staging: &Path,
) -> Result<u64> {
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent).context("staging dir")?;
    }
    let mut file =
        std::fs::File::create(staging).with_context(|| format!("create {}", staging.display()))?;
    let (bytes, frames) = pump_recv(session, &mut file, DEFAULT_WINDOW)?;
    file.sync_all().context("staging sync")?;
    drop(file);
    check_artifact_file(manifest, staging).context("artifact verification")?;
    eprintln!(
        "update: received {} bytes in {} frames for {}",
        bytes, frames, manifest.version
    );
    Ok(bytes)
}

/// One signed line back on the update socket — the frame stream is
/// over; the wire is line mode again (WP-U1 final-ack law).
fn write_line(sock: &mut std::net::TcpStream, secret: &Secret, body: &str) -> Result<()> {
    use ouro_cluster::transport::auth;
    let line = auth::sign_line(secret, 1, body);
    sock.write_all(line.as_bytes())?;
    sock.write_all(b"\n")?;
    sock.flush()?;
    Ok(())
}

fn receipt(manifest: &Manifest, status: &str, output: &str) -> String {
    serde_json::json!({
        "task_id": manifest.version,
        "status": status,
        "output": output,
        "elapsed_ms": 0,
        "peak_watts": 0,
    })
    .to_string()
}

/// The whole update transaction on one connection: verify manifest →
/// receive artifact frames → install/reflash → receipt. Agent
/// artifacts exec in place before returning (this thread is replaced).
pub fn handle_update(
    secret: Secret,
    sock: std::net::TcpStream,
    expect: Artifact,
    wire: String,
) -> Result<()> {
    let manifest = match verify_manifest(&wire, expect) {
        Ok(m) => m,
        Err(e) => {
            let mut sock = sock;
            let _ = write_line(&mut sock, &secret, &format!("err update: {e:#}"));
            return Err(e);
        }
    };

    let staging = match expect {
        Artifact::Agent => run_dir().join("updates/agent.new"),
        Artifact::Image => ouro_mount().join("ouro-update.iso"),
    };
    let mut session = FrameSession::new(sock, secret);
    if let Err(e) = receive_artifact(&mut session, &manifest, &staging) {
        let _ = write_line(session.get_mut(), &secret, &format!("err update: {e:#}"));
        return Err(e);
    }

    match expect {
        Artifact::Agent => {
            if let Err(e) = install_agent(&staging, &wire) {
                let _ = write_line(
                    session.get_mut(),
                    &secret,
                    &format!("err update: {e:#}"),
                );
                return Err(e);
            }
            write_line(
                session.get_mut(),
                &secret,
                &receipt(&manifest, "Success", "installed; exec-ing live agent"),
            )?;
            drop(session);
            exec_installed_agent();
        }
        Artifact::Image => {
            let msg = match self_reflash(&staging, &manifest) {
                Ok(m) => m,
                Err(e) => {
                    let _ = write_line(
                        session.get_mut(),
                        &secret,
                        &format!("err update: {e:#}"),
                    );
                    return Err(e);
                }
            };
            write_line(
                session.get_mut(),
                &secret,
                &receipt(&manifest, "Success", &msg),
            )?;
            drop(session);
            reboot()?;
        }
    }
    Ok(())
}

/// Replace this process with the freshly installed agent — same argv,
/// same environment (the shim's exports ride along). On exec failure
/// the caller exits; getty respawns the baked-in binary, whose
/// boot_check then execs the live agent itself (self-healing).
fn exec_installed_agent() {
    let (live, _, _) = live_paths();
    let args: Vec<String> = std::env::args().skip(1).collect();
    eprintln!("update: exec {}", live.display());
    use std::os::unix::process::CommandExt;
    // exec replaces the process on success and only ever returns on
    // failure (the return value IS the error, not a Result).
    let err = std::process::Command::new(&live).args(&args).exec();
    eprintln!("update: exec failed: {err}");
    std::process::exit(1);
}

/// Install a verified agent artifact: the tmpfs slot for this session,
/// the OURO partition for persistence, the signed manifest beside it.
pub fn install_agent(staging: &Path, manifest_wire: &str) -> Result<()> {
    let (live, manifest_path, counter_path) = live_paths();
    std::fs::copy(staging, &live).with_context(|| format!("install {}", live.display()))?;
    std::fs::copy(staging, ouro_mount().join("agent-live"))
        .with_context(|| format!("persist {}", ouro_mount().display()))?;
    std::fs::write(&manifest_path, manifest_wire).context("persist manifest")?;
    std::fs::write(&counter_path, "0").context("reset counter")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&live, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("chmod {}", live.display()))?;
    }
    Ok(())
}

/// Parent disk of a partition name: `sdc3` → `sdc`, `nvme0n1p3` →
/// `nvme0n1`, `mmcblk0p2` → `mmcblk0`. One digit-group, optional p.
pub fn parent_disk(part: &str) -> Result<String> {
    let base = part.trim_end_matches(|c: char| c.is_ascii_digit());
    if base == part || base.is_empty() {
        bail!("not a partition name: {part:?}");
    }
    let base = base.strip_suffix('p').unwrap_or(base);
    if base.is_empty() {
        bail!("not a partition name: {part:?}");
    }
    Ok(base.to_string())
}

/// Parse the OURO partition's start sector out of `sfdisk -d` output.
pub fn start_sector_from_sfdisk(out: &str, part_name: &str) -> Option<u64> {
    for line in out.lines() {
        let line = line.trim();
        let Some(idx) = line.find("start=") else { continue };
        if !line.starts_with(part_name) && !line.contains(&format!("/{part_name} ")) {
            continue;
        }
        let rest = &line[idx + "start=".len()..];
        let num: String = rest
            .chars()
            .skip_while(|c| c.is_whitespace() || *c == '=')
            .take_while(|c| c.is_ascii_digit())
            .collect();
        return num.parse().ok();
    }
    None
}

/// The write-guard: the ISO must end before the OURO partition
/// begins — a self-reflash never touches the anchor.
pub fn iso_fits(start_sector: u64, iso_size: u64) -> bool {
    iso_size <= start_sector.saturating_mul(512)
}

/// The boot medium: whatever disk carries the OURO partition.
pub fn resolve_boot_device() -> Result<(PathBuf, String)> {
    let target = std::fs::read_link(OURO_PARTITION)
        .with_context(|| format!("readlink {OURO_PARTITION} (not mounted/no label?)"))?;
    let part = target
        .file_name()
        .and_then(|n| n.to_str())
        .context("partition name")?
        .to_string();
    let disk = parent_disk(&part)?;
    Ok((Path::new("/dev").join(&disk), part))
}

fn on_battery() -> bool {
    (0..4).any(|i| {
        std::fs::read_to_string(format!("/sys/class/power_supply/BAT{i}/status"))
            .map(|s| s.trim() == "Discharging")
            .unwrap_or(false)
    })
}

/// Full self-reflash: staging ISO → own boot medium → readback verify
/// → clean reboot. Every rail is load-bearing (UPDATE_ROADMAP §rails).
pub fn self_reflash(staging: &Path, manifest: &Manifest) -> Result<String> {
    if on_battery() {
        bail!("refusing reflash on battery power (Art. 4: the tail stays healthy)");
    }
    let (device, part) = resolve_boot_device()?;
    let sfdisk = std::process::Command::new("sfdisk")
        .arg("-d")
        .arg(&device)
        .output()
        .context("sfdisk -d")?;
    let out = String::from_utf8_lossy(&sfdisk.stdout).to_string();
    let start = start_sector_from_sfdisk(&out, &part)
        .with_context(|| format!("OURO start sector not found for {part}"))?;
    let (iso_size, _) = sha256_file(staging)?;
    if !iso_fits(start, iso_size) {
        bail!(
            "ISO {} bytes overruns OURO partition start sector {start} — reflashing would eat the anchor",
            iso_size
        );
    }
    // Write [0, iso_size): RAM-resident root means nothing else holds
    // the boot region open; the OURO range is never touched.
    {
        let mut dev = std::fs::OpenOptions::new()
            .write(true)
            .open(&device)
            .with_context(|| format!("open {} writable", device.display()))?;
        let mut src = std::fs::File::open(staging)?;
        std::io::copy(&mut src, &mut dev).context("write ISO to boot medium")?;
        dev.sync_all().context("sync boot medium")?;
    }
    // Readback law — the same one flash.sh obeys on the head.
    {
        let mut dev = std::fs::File::open(&device).context("reopen boot medium")?;
        let mut hasher = Sha256::new();
        let mut remaining = iso_size;
        let mut buf = vec![0u8; 1024 * 1024];
        while remaining > 0 {
            let want = buf.len().min(remaining as usize);
            let n = dev.read(&mut buf[..want])?;
            if n == 0 {
                bail!("short readback (got {}, wanted {iso_size})", iso_size - remaining);
            }
            hasher.update(&buf[..n]);
            remaining -= n as u64;
        }
        let digest = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
        if digest != manifest.sha256 {
            bail!("readback sha256 mismatch — boot medium not touched by reboot, staging retained");
        }
    }
    let _ = std::fs::remove_file(staging);
    Ok(format!("reflashed {} ({} bytes), rebooting into {}", device.display(), iso_size, manifest.version))
}

/// Clean reboot (polkit rule ships in the image for the ouro user).
pub fn reboot() -> Result<()> {
    std::process::Command::new("systemctl")
        .args(["reboot"])
        .spawn()
        .context("spawn systemctl reboot")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parent_disk() {
        assert_eq!(parent_disk("sdc3").unwrap(), "sdc");
        assert_eq!(parent_disk("nvme0n1p3").unwrap(), "nvme0n1");
        assert_eq!(parent_disk("mmcblk0p2").unwrap(), "mmcblk0");
        assert_eq!(parent_disk("sda12").unwrap(), "sda");
        assert!(parent_disk("sdc").is_err(), "bare disk is not a partition");
        assert!(parent_disk("").is_err());
    }

    #[test]
    fn test_start_sector_parse() {
        let sfdisk = "label: dos\nlabel-id: 0x1234\ndevice: /dev/sdc\nunit: sectors\n\n/dev/sdc1 : start= 2048, size= 8386560, type=7\n/dev/sdc3 : start= 17154048, size= 58060800, type=c\n";
        assert_eq!(start_sector_from_sfdisk(sfdisk, "sdc3"), Some(17154048));
        assert_eq!(start_sector_from_sfdisk(sfdisk, "sdc1"), Some(2048));
        assert_eq!(start_sector_from_sfdisk(sfdisk, "sdc9"), None);
    }

    #[test]
    fn test_iso_fits_guard() {
        // OURO at sector 17154048 → byte 8777675776 (≈8.2GiB)
        assert!(iso_fits(17154048, 700 * 1024 * 1024));
        assert!(!iso_fits(17154048, 9 * 1024 * 1024 * 1024));
        // degenerate: start 0 accepts nothing
        assert!(!iso_fits(0, 1));
    }

    #[test]
    fn test_boot_decision() {
        use BootAction::*;
        assert_eq!(boot_decision(false, 0, true), UseBakedIn);
        assert_eq!(boot_decision(true, 0, true), ExecLive);
        assert_eq!(boot_decision(true, 1, true), ExecLive);
        assert_eq!(boot_decision(true, 2, true), ExecLive);
        assert_eq!(boot_decision(true, 3, true), DiscardLive, "three strikes");
        assert_eq!(boot_decision(true, 0, false), DiscardLive, "bad signature");
    }

    #[test]
    fn test_verify_manifest_structure() {
        // no pubkey file on dev machines: the error must be honest
        let err = verify_manifest("{}", Artifact::Agent).unwrap_err().to_string();
        assert!(err.contains("update.pub") || err.contains("read"), "got: {err}");
    }
}
