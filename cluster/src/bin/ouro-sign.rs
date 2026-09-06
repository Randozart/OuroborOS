//! ouro-sign — WP-UPDATE key ceremony and manifest signing (WP-U2).
//!
//! The private key NEVER leaves the head: `keys/update.signing.key`
//! (hex seed, 0600, gitignored — never in enroll/, which gets flashed
//! to sticks). The public key is committed and baked into every image
//! at /etc/ouro/update.pub. Tails verify; only the head signs.
//!
//! Usage:
//!   ouro-sign keygen [DIR]                      default ./keys
//!   ouro-sign sign <manifest.json> [-k KEY]     signed JSON on stdout
//!   ouro-sign verify <signed.json> [-p PUB] [artifact]
use anyhow::{bail, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use ouro_cluster::update::{self, Artifact, Manifest};
use rand_core::OsRng;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        bail!("usage: ouro-sign <keygen|sign|verify> ...");
    };
    match cmd.as_str() {
        "keygen" => keygen(&dir_arg(&args)),
        "sign" => sign_cmd(&args[1..]),
        "verify" => verify_cmd(&args[1..]),
        other => bail!("unknown command {other:?} — use keygen|sign|verify"),
    }
}

fn dir_arg(args: &[String]) -> PathBuf {
    args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("keys"))
}

fn key_path(dir: &Path) -> PathBuf {
    dir.join("update.signing.key")
}

fn pub_path(dir: &Path) -> PathBuf {
    dir.join("update.signing.pub")
}

fn load_signing_key(path: &PathBuf) -> Result<SigningKey> {
    let hex = std::fs::read_to_string(path)
        .with_context(|| format!("read signing key {}", path.display()))?;
    let hex = hex.trim();
    let mut seed = [0u8; 32];
    if hex.len() != 64 {
        bail!("signing key must be 64 hex chars (32 bytes), got {}", hex.len());
    }
    for i in 0..32 {
        seed[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).context("key hex")?;
    }
    Ok(SigningKey::from_bytes(&seed))
}

fn load_verifying_key(path: &PathBuf) -> Result<VerifyingKey> {
    let hex = std::fs::read_to_string(path)
        .with_context(|| format!("read public key {}", path.display()))?;
    let hex = hex.trim();
    let mut bytes = [0u8; 32];
    if hex.len() != 64 {
        bail!("public key must be 64 hex chars (32 bytes), got {}", hex.len());
    }
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).context("pub hex")?;
    }
    VerifyingKey::from_bytes(&bytes).context("public key structure")
}

/// Generate the ed25519 signing pair. Refuses to overwrite an existing
/// key — rotation is a ceremony, not an accident.
fn keygen(dir: &PathBuf) -> Result<()> {
    let kpath = key_path(dir);
    if kpath.exists() {
        bail!("{} already exists — refusing to overwrite", kpath.display());
    }
    let signing = SigningKey::generate(&mut OsRng);
    std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let seed_hex: String = signing
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&kpath, seed_hex).with_context(|| format!("write {}", kpath.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&kpath, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", kpath.display()))?;
    }
    let pub_hex: String = signing
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(pub_path(dir), pub_hex).context("write public key")?;
    println!("signing key: {}", kpath.display());
    println!("public key:  {}", pub_path(dir).display());
    println!("bake the public key into the image at /etc/ouro/update.pub");
    Ok(())
}

fn sign_cmd(args: &[String]) -> Result<()> {
    let mut manifest_path = None;
    let mut key_file = key_path(&PathBuf::from("keys"));
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-k" | "--key" => {
                key_file = PathBuf::from(it.next().context("-k needs a path")?);
            }
            other if manifest_path.is_none() => manifest_path = Some(other.to_string()),
            other => bail!("unexpected arg {other:?}"),
        }
    }
    let Some(path) = manifest_path else {
        bail!("usage: ouro-sign sign <manifest.json> [-k KEY]");
    };
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    // Accept either a bare manifest (no sig) or one with a stale sig —
    // the signature is always recomputed over the canonical form.
    let value: serde_json::Value = serde_json::from_str(&raw).context("manifest JSON")?;
    let manifest: Manifest =
        serde_json::from_value(value).context("manifest fields")?;
    if manifest.kind != update::KIND {
        bail!("manifest kind {:?} != {}", manifest.kind, update::KIND);
    }
    let signing = load_signing_key(&key_file)?;
    print!("{}", update::sign(&manifest, &signing)?);
    std::io::stdout().flush().ok();
    Ok(())
}

fn verify_cmd(args: &[String]) -> Result<()> {
    let mut wire_path = None;
    let mut pub_file = pub_path(&PathBuf::from("keys"));
    let mut artifact_path = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" | "--pub" => {
                pub_file = PathBuf::from(it.next().context("-p needs a path")?);
            }
            other if wire_path.is_none() => wire_path = Some(other.to_string()),
            other if artifact_path.is_none() => artifact_path = Some(other.to_string()),
            other => bail!("unexpected arg {other:?}"),
        }
    }
    let Some(path) = wire_path else {
        bail!("usage: ouro-sign verify <signed.json> [-p PUB] [artifact]");
    };
    let wire = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let pubkey = load_verifying_key(&pub_file)?;
    let manifest = update::verify_signature(&wire, &pubkey)?;
    println!("signature: OK (key_id {}, artifact {}, version {})",
        manifest.key_id, manifest.artifact, manifest.version);
    if let Some(apath) = artifact_path {
        let bytes = std::fs::read(&apath).with_context(|| format!("read {apath}"))?;
        let expect = Artifact::parse(&manifest.artifact)?;
        update::check_artifact(&manifest, &bytes, expect)?;
        println!("artifact:  OK ({} bytes, sha256 {})", bytes.len(), manifest.sha256);
    }
    Ok(())
}
