//! HMAC-SHA256 line auth (WP2, PLAN §9.3 core).
//!
//! Frame-agnostic by design: `tag` + `verify` are pure functions over
//! `(secret, seq, payload)` and drop into the OURO frame trailer
//! unchanged when the wire moves to frames (L2, EtherType 0x88B5).
//!
//! Wire encoding on the current newline-JSON transport is a flat prefix,
//! body verbatim (no envelope — an envelope would re-serialize every
//! 20–50KB ACTS payload per hop):
//!
//! ```text
//! <seq> <64-hex-tag> <body...>\n
//! ```
//!
//! Both directions are signed. Auth failure → terse rejection, no detail.
//! Limitation (Art. 10 honesty): connect-per-request wires get integrity
//! and authenticity from this; seq correlation is not server-side
//! anti-replay until connections persist (frame upgrade path).
use anyhow::{bail, Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::pipeline::{from_hex, to_hex};

type HmacSha256 = Hmac<Sha256>;

/// Shared secret: 32 bytes, provisioned out-of-band (manual copy for
/// node #1; R2_BRINGUP.md §8).
pub type Secret = [u8; 32];

/// Environment variable holding the path to the hex-encoded secret file.
pub const SECRET_FILE_ENV: &str = "OURO_SECRET_FILE";

/// HMAC-SHA256 over `seq || payload`. Seq rides big-endian so the same
/// (seq, payload) pair hashes identically everywhere.
pub fn tag(secret: &Secret, seq: u64, payload: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(secret).expect("32-byte key valid for HMAC-SHA256");
    mac.update(&seq.to_be_bytes());
    mac.update(payload);
    mac.finalize().into_bytes().into()
}

/// Constant-time tag comparison.
pub fn verify(secret: &Secret, seq: u64, payload: &[u8], expected: &[u8; 32]) -> bool {
    let computed = tag(secret, seq, payload);
    let mut diff = 0u8;
    for (a, b) in computed.iter().zip(expected.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Encode one authenticated line: `seq tag body`.
pub fn sign_line(secret: &Secret, seq: u64, body: &str) -> String {
    format!("{} {} {}", seq, to_hex(&tag(secret, seq, body.as_bytes())), body)
}

/// Open one authenticated line. Returns `(seq, body)` on a valid tag;
/// any structural or tag failure is the same opaque error (no oracle).
pub fn open_line<'a>(secret: &Secret, line: &'a str) -> Result<(u64, &'a str)> {
    let mut parts = line.splitn(3, ' ');
    let (seq_s, tag_s, body) = match (parts.next(), parts.next(), parts.next()) {
        (Some(s), Some(t), Some(b)) => (s, t, b),
        _ => bail!("auth"),
    };
    let seq: u64 = seq_s.parse().map_err(|_| anyhow::anyhow!("auth"))?;
    let expected: [u8; 32] = from_hex(tag_s)
        .ok()
        .and_then(|v| <[u8; 32]>::try_from(v).ok())
        .ok_or_else(|| anyhow::anyhow!("auth"))?;
    if !verify(secret, seq, body.as_bytes(), &expected) {
        bail!("auth");
    }
    Ok((seq, body))
}

/// Load the 32-byte secret from `OURO_SECRET_FILE` (64 hex chars).
/// Mandatory: callers refuse to start without it (no bypass flag).
pub fn secret_from_env() -> Result<Secret> {
    let path = std::env::var(SECRET_FILE_ENV)
        .with_context(|| format!("{} not set — refusing unauthenticated wire", SECRET_FILE_ENV))?;
    secret_from_file(&path)
}

/// Read + parse a secret file.
pub fn secret_from_file(path: &str) -> Result<Secret> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read secret file {}", path))?;
    let hex = text.trim();
    let bytes = from_hex(hex).context("parse secret hex")?;
    let secret: Secret = bytes
        .try_into()
        .map_err(|v: Vec<u8>| anyhow::anyhow!("secret must be 32 bytes, got {}", v.len()))?;
    Ok(secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: Secret = [7u8; 32];
    const BODY: &str = r#"{"id":"t1","name":"echo","payload":"hello"}"#;

    #[test]
    fn test_tag_deterministic_and_seq_sensitive() {
        let a = tag(&KEY, 1, BODY.as_bytes());
        let b = tag(&KEY, 1, BODY.as_bytes());
        let c = tag(&KEY, 2, BODY.as_bytes());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_verify_key_sensitive() {
        let t = tag(&KEY, 1, BODY.as_bytes());
        assert!(verify(&KEY, 1, BODY.as_bytes(), &t));
        let other: Secret = [8u8; 32];
        assert!(!verify(&other, 1, BODY.as_bytes(), &t));
        assert!(!verify(&KEY, 2, BODY.as_bytes(), &t));
        assert!(!verify(&KEY, 1, b"tampered", &t));
    }

    #[test]
    fn test_sign_open_roundtrip() {
        let line = sign_line(&KEY, 42, BODY);
        let (seq, body) = open_line(&KEY, &line).unwrap();
        assert_eq!(seq, 42);
        assert_eq!(body, BODY);
    }

    #[test]
    fn test_open_line_tamper_rejected() {
        let line = sign_line(&KEY, 1, BODY);
        let tampered = line.replace("hello", "EVIL");
        assert!(open_line(&KEY, &tampered).is_err());
    }

    #[test]
    fn test_open_line_wrong_key_rejected() {
        let line = sign_line(&KEY, 1, BODY);
        let other: Secret = [9u8; 32];
        assert!(open_line(&other, &line).is_err());
    }

    #[test]
    fn test_open_line_structural_failures_opaque() {
        for bad in ["", "1", "1 aa bb", "x <tag> body", "1 zz body"] {
            assert!(open_line(&KEY, bad).is_err(), "case {:?}", bad);
        }
        let malformed = format!("1 {}", to_hex(&[0u8; 32]));
        assert!(open_line(&KEY, &malformed).is_err());
    }

    #[test]
    fn test_secret_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ouro_auth_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secret.hex");
        std::fs::write(&path, to_hex(&KEY)).unwrap();
        assert_eq!(secret_from_file(path.to_str().unwrap()).unwrap(), KEY);
        std::fs::write(&path, to_hex(&KEY) + "\n").unwrap();
        assert_eq!(secret_from_file(path.to_str().unwrap()).unwrap(), KEY);
        std::fs::write(&path, "abcd").unwrap();
        assert!(secret_from_file(path.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
