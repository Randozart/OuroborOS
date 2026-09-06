//! Update manifest contract (WP-U2, UPDATE_ROADMAP).
//!
//! Two trust planes (UPDATE_ROADMAP §trust): the HMAC wire secret
//! authorizes transport, ed25519 authorizes content. A tail verifies
//! the signature over the CANONICAL manifest bytes regardless of how
//! the artifact arrived — trust follows the signature, not the channel.
//!
//! Canonical form: the manifest serialized compactly in struct field
//! order (the order documented in UPDATE_ROADMAP §manifest). The
//! signature covers exactly those bytes; the wire form may vary in
//! whitespace or key order, never in meaning. `verify_signature`
//! re-canonicalizes before verifying, so transport form is irrelevant.
use anyhow::{bail, Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Contract version of the manifest format itself.
pub const KIND: &str = "ouro-update/1";

/// Wire/agent protocol level required to interpret this manifest.
pub const PROTOCOL: u32 = 1;

/// What is being pushed. An `agent` push hot-swaps the binary in place
/// (closure-match rule applies); an `image` push rewrites the tail's
/// own boot medium and reboots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artifact {
    Agent,
    Image,
}

impl Artifact {
    pub fn as_str(&self) -> &'static str {
        match self {
            Artifact::Agent => "agent",
            Artifact::Image => "image",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "agent" => Ok(Artifact::Agent),
            "image" => Ok(Artifact::Image),
            other => bail!("unknown artifact kind {other:?}"),
        }
    }
}

/// The update manifest. FIELD ORDER IS THE CANONICAL FORM — the
/// signature covers the compact serialization in exactly this order;
/// reordering fields invalidates every existing signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub kind: String,
    pub key_id: String,
    pub artifact: String,
    pub version: String,
    pub image_rev: String,
    pub size: u64,
    pub sha256: String,
    pub min_agent_protocol: u32,
    pub reboot: bool,
    pub created_utc: String,
}

impl Manifest {
    pub fn new(key_id: &str, artifact: Artifact, version: &str, image_rev: &str) -> Self {
        Self {
            kind: KIND.to_string(),
            key_id: key_id.to_string(),
            artifact: artifact.as_str().to_string(),
            version: version.to_string(),
            image_rev: image_rev.to_string(),
            size: 0,
            sha256: String::new(),
            min_agent_protocol: PROTOCOL,
            reboot: matches!(artifact, Artifact::Image),
            created_utc: String::new(),
        }
    }

    /// The bytes the signature covers: compact JSON, struct field
    /// order, sig field excluded (it cannot cover itself).
    pub fn canonical(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).context("canonical serialization")
    }
}

/// Sign a manifest; returns the wire JSON (manifest + `sig`, hex).
/// The wire form is sorted-key compact JSON — valid JSON, not the
/// canonical order; only the canonical bytes are signed.
pub fn sign(manifest: &Manifest, signing: &SigningKey) -> Result<String> {
    let bytes = manifest.canonical()?;
    let sig = signing.sign(&bytes);
    let mut value = serde_json::to_value(manifest).context("manifest to value")?;
    value
        .as_object_mut()
        .context("manifest is an object")?
        .insert("sig".into(), serde_json::Value::String(hex(&sig.to_bytes())));
    serde_json::to_string(&value).context("wire serialization")
}

/// Verify the signature on a wire JSON manifest. Re-canonicalizes the
/// parsed manifest (transport form is irrelevant) and verifies
/// strictly against `pubkey`. Returns the manifest on success.
pub fn verify_signature(wire: &str, pubkey: &VerifyingKey) -> Result<Manifest> {
    let value: serde_json::Value =
        serde_json::from_str(wire).context("manifest JSON parse")?;
    let obj = value.as_object().context("manifest is an object")?;
    let sig_hex = obj
        .get("sig")
        .and_then(|v| v.as_str())
        .context("manifest missing sig")?;
    let sig_bytes = from_hex(sig_hex).context("sig hex")?;
    let sig = Signature::from_slice(&sig_bytes).context("sig structure")?;
    let manifest: Manifest =
        serde_json::from_value(value.clone()).context("manifest fields")?;
    if manifest.kind != KIND {
        bail!("manifest kind {:?} != {}", manifest.kind, KIND);
    }
    pubkey
        .verify_strict(&manifest.canonical()?, &sig)
        .context("manifest signature")?;
    Ok(manifest)
}

/// Artifact-level checks: kind expected, protocol compatible, size and
/// sha256 match the bytes actually received. Signature verification is
/// a precondition — this never runs on an unverified manifest.
pub fn check_artifact(manifest: &Manifest, bytes: &[u8], expect: Artifact) -> Result<()> {
    let artifact = Artifact::parse(&manifest.artifact)?;
    if artifact != expect {
        bail!("manifest artifact {} != expected {}", artifact.as_str(), expect.as_str());
    }
    if manifest.min_agent_protocol > PROTOCOL {
        bail!(
            "manifest needs protocol {} > ours {}",
            manifest.min_agent_protocol,
            PROTOCOL
        );
    }
    if manifest.size != bytes.len() as u64 {
        bail!("manifest size {} != received {}", manifest.size, bytes.len());
    }
    let digest = hex(&Sha256::digest(bytes));
    if manifest.sha256 != digest {
        bail!("artifact sha256 mismatch (manifest {} != actual {digest})", manifest.sha256);
    }
    Ok(())
}

/// Build a manifest for concrete artifact bytes (fills size + sha256).
pub fn with_artifact(mut manifest: Manifest, bytes: &[u8], created_utc: &str) -> Manifest {
    manifest.size = bytes.len() as u64;
    manifest.sha256 = hex(&Sha256::digest(bytes));
    manifest.created_utc = created_utc.to_string();
    manifest
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        bail!("odd hex length");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| anyhow::anyhow!("hex: {e}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[42u8; 32])
    }

    fn test_manifest() -> Manifest {
        let bytes = b"artifact-bytes-for-testing";
        with_artifact(
            Manifest::new("test-key", Artifact::Agent, "git:abc1234", "abc1234"),
            bytes,
            "2026-09-06T12:00:00Z",
        )
    }

    #[test]
    fn test_canonical_deterministic() {
        let m = test_manifest();
        let a = m.canonical().unwrap();
        let b = m.canonical().unwrap();
        assert_eq!(a, b);
        // compact: no whitespace
        assert!(!a.contains(&b' '));
        // field order fixed: kind first, created_utc last
        let s = String::from_utf8(a).unwrap();
        assert!(s.starts_with(r#"{"kind":"ouro-update/1""#));
        assert!(s.ends_with(r#""created_utc":"2026-09-06T12:00:00Z"}"#));
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let key = test_key();
        let wire = sign(&test_manifest(), &key).unwrap();
        let verified = verify_signature(&wire, &key.verifying_key()).unwrap();
        assert_eq!(verified, test_manifest());
    }

    #[test]
    fn test_wire_form_variations_verify() {
        // the signature covers canonical bytes; wire form may differ
        let key = test_key();
        let wire = sign(&test_manifest(), &key).unwrap();
        // pretty-printed wire form with reordered keys still verifies
        let value: serde_json::Value = serde_json::from_str(&wire).unwrap();
        let pretty = serde_json::to_string_pretty(&value).unwrap();
        assert!(verify_signature(&pretty, &key.verifying_key()).is_ok());
    }

    #[test]
    fn test_tamper_rejected() {
        let key = test_key();
        let mut value: serde_json::Value =
            serde_json::from_str(&sign(&test_manifest(), &key).unwrap()).unwrap();
        // flip version
        value["version"] = serde_json::Value::String("git:evil000".into());
        let wire = serde_json::to_string(&value).unwrap();
        assert!(verify_signature(&wire, &key.verifying_key()).is_err());
    }

    #[test]
    fn test_wrong_key_rejected() {
        let wire = sign(&test_manifest(), &test_key()).unwrap();
        let other = SigningKey::from_bytes(&[43u8; 32]);
        assert!(verify_signature(&wire, &other.verifying_key()).is_err());
    }

    #[test]
    fn test_missing_sig_rejected() {
        let value = serde_json::to_value(test_manifest()).unwrap();
        let wire = serde_json::to_string(&value).unwrap();
        assert!(verify_signature(&wire, &test_key().verifying_key()).is_err());
    }

    #[test]
    fn test_artifact_checks() {
        let bytes = b"artifact-bytes-for-testing";
        let m = test_manifest();
        check_artifact(&m, bytes, Artifact::Agent).unwrap();

        // sha mismatch
        assert!(check_artifact(&m, b"tampered-bytes!!", Artifact::Agent).is_err());
        // size mismatch
        assert!(check_artifact(&m, b"short", Artifact::Agent).is_err());
        // kind mismatch
        assert!(check_artifact(&m, bytes, Artifact::Image).is_err());
        // protocol too new
        let mut future = m.clone();
        future.min_agent_protocol = PROTOCOL + 1;
        assert!(check_artifact(&future, bytes, Artifact::Agent).is_err());
    }

    #[test]
    fn test_image_manifest_reboots() {
        let m = Manifest::new("k", Artifact::Image, "git:a", "a");
        assert!(m.reboot);
        let m = Manifest::new("k", Artifact::Agent, "git:a", "a");
        assert!(!m.reboot);
    }
}
