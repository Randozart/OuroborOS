//! Bonsai tokenizer — GPT-2 byte-level BPE with the qwen35 pretokenizer
//! (docs/BONSAI_TOKENIZER.md). Vocabulary is loaded from `tokenizer.json`
//! dumped from the GGUF by `tools/dump_tokenizer.py`; the Rust engine never
//! parses GGUF for vocab.
//!
//! Gates are oracle-anchored: `tokenize("Hello") == [9419]` and the
//! detokenized greedy stream `, I'm a student in the University` were both
//! emitted by the PrismML fork in verified capture runs.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// The qwen35 pretokenizer, verbatim from the fork (llama-vocab.cpp
/// LLAMA_VOCAB_PRE_TYPE_QWEN35). Qwen2 with letters widened to
/// letters+marks. Contains `(?!\S)` — needs fancy-regex.
pub const QWEN35_PRE_PATTERN: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#[derive(Debug, Clone, Deserialize)]
struct VocabFile {
    /// always "gpt2" (byte-level BPE) for these checkpoints
    #[allow(dead_code)]
    model: String,
    /// pretokenizer name; asserted-compatible at load
    pre: String,
    tokens: Vec<String>,
    token_types: Vec<i32>,
    merges: Vec<String>,
    eos: Option<u32>,
    /// pad == bos for Bonsai; unused at completion time
    #[allow(dead_code)]
    bos: Option<u32>,
    #[allow(dead_code)]
    pad: Option<u32>,
    /// false for Bonsai (no BOS prepend)
    #[allow(dead_code)]
    add_bos: bool,
}

/// token_type codes (llama.cpp convention).
const TYPE_NORMAL: i32 = 1;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_UNUSED: i32 = 5;

/// Byte-level BPE tokenizer for qwen35 checkpoints.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab_file: VocabFile,
    id_by_token: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pre_regex: fancy_regex::Regex,
}

impl Tokenizer {
    /// Load from a shard dir's `tokenizer.json`.
    pub fn load_dir(dir: &str) -> Result<Self> {
        let path = format!("{dir}/tokenizer.json");
        let text = std::fs::read_to_string(&path).context(format!("reading {path}"))?;
        let vf: VocabFile = serde_json::from_str(&text).context("parsing tokenizer.json")?;
        if vf.tokens.len() != vf.token_types.len() {
            bail!("tokens/token_type arity mismatch");
        }
        if vf.pre != "qwen35" {
            bail!("unsupported pretokenizer {:?} (want qwen35)", vf.pre);
        }
        let mut id_by_token = HashMap::with_capacity(vf.tokens.len());
        for (i, t) in vf.tokens.iter().enumerate() {
            id_by_token.insert(t.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(vf.merges.len());
        for (i, m) in vf.merges.iter().enumerate() {
            let (a, b) = m
                .split_once(' ')
                .ok_or_else(|| anyhow::anyhow!("merge {i} has no space: {m:?}"))?;
            ranks.insert((a.to_string(), b.to_string()), i as u32);
        }
        let byte_encoder = build_byte_encoder();
        let byte_decoder: HashMap<char, u8> =
            byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();
        Ok(Self {
            pre_regex: fancy_regex::Regex::new(QWEN35_PRE_PATTERN)
                .context("compiling qwen35 pre regex")?,
            vocab_file: vf,
            id_by_token,
            ranks,
            byte_encoder,
            byte_decoder,
        })
    }

    pub fn eos(&self) -> Option<u32> {
        self.vocab_file.eos
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab_file.tokens.len()
    }

    /// GPT-2 byte-level alphabet: printable bytes map to themselves; the
    /// rest map to 256+n in increasing byte order (the HF tokenizer.net
    /// construction, byte-for-byte).
    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>> {
        // raw text bytes -> escaped alphabet string
        let escaped: String = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[&b])
            .collect();
        if escaped.is_empty() {
            return Ok(Vec::new());
        }
        // single symbol is already a token candidate — fast path
        if escaped.chars().count() == 1 {
            if let Some(&id) = self.id_by_token.get(&escaped) {
                return Ok(vec![id]);
            }
        }
        let mut parts: Vec<String> = escaped.chars().map(|c| c.to_string()).collect();
        // classic greedy lowest-rank merge
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        let mut out = Vec::with_capacity(parts.len());
        for p in &parts {
            let Some(&id) = self.id_by_token.get(p) else {
                bail!("piece {p:?} not in vocab (from {piece:?})");
            };
            out.push(id);
        }
        Ok(out)
    }

    /// Encode text to token ids. Special tokens in the text are NOT
    /// matched specially (base-model completion; no chat template yet).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut last = 0usize;
        for m in self.pre_regex.find_iter(text) {
            let m = m.map_err(|e| anyhow::anyhow!("pre-regex: {e}"))?;
            if m.start() > last {
                // defensive: regex must cover the input; never lose text
                ids.extend(self.encode_piece(&text[last..m.start()])?);
            }
            ids.extend(self.encode_piece(m.as_str())?);
            last = m.end();
        }
        if last < text.len() {
            ids.extend(self.encode_piece(&text[last..])?);
        }
        Ok(ids)
    }

    /// Token string bytes for decode; control/unused tokens decode empty.
    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.vocab_file.tokens.get(id as usize) else {
            return Vec::new();
        };
        let ty = self.vocab_file.token_types.get(id as usize).copied().unwrap_or(TYPE_NORMAL);
        if ty == TYPE_CONTROL || ty == TYPE_UNUSED {
            return Vec::new();
        }
        let _ = TYPE_USER_DEFINED;
        t.chars().map(|c| self.byte_decoder[&c]).collect()
    }

    /// Decode ids to text (full-string convenience).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut d = StreamingDecoder::new(self.clone_handle()?);
        let mut out = String::new();
        for &id in ids {
            out.push_str(&d.push(id));
        }
        out.push_str(&d.finish());
        Ok(out)
    }

    fn clone_handle(&self) -> Result<Tokenizer> {
        // cheap-enough full clone; the streaming decoder owns its source
        Self::load_from_parts(
            self.vocab_file.clone(),
            self.id_by_token.clone(),
            self.ranks.clone(),
            self.byte_encoder.clone(),
            self.byte_decoder.clone(),
            self.pre_regex.clone(),
        )
    }
}

impl Tokenizer {
    fn load_from_parts(
        vocab_file: VocabFile,
        id_by_token: HashMap<String, u32>,
        ranks: HashMap<(String, String), u32>,
        byte_encoder: HashMap<u8, char>,
        byte_decoder: HashMap<char, u8>,
        pre_regex: fancy_regex::Regex,
    ) -> Result<Self> {
        Ok(Self {
            vocab_file,
            id_by_token,
            ranks,
            byte_encoder,
            byte_decoder,
            pre_regex,
        })
    }
}

/// Incremental detokenizer: buffers bytes until a UTF-8 boundary, so a
/// multi-byte char split across tokens never emits replacement chars.
pub struct StreamingDecoder {
    tok: Tokenizer,
    buf: Vec<u8>,
}

impl StreamingDecoder {
    pub fn new(tok: Tokenizer) -> Self {
        Self { tok, buf: Vec::new() }
    }

    /// Feed one token; returns newly-completed text (possibly empty).
    pub fn push(&mut self, id: u32) -> String {
        self.buf.extend(self.tok.token_bytes(id));
        self.drain_complete()
    }

    /// Flush any buffered tail (call at end of generation).
    pub fn finish(mut self) -> String {
        let out = String::from_utf8_lossy(&self.buf).to_string();
        self.buf.clear();
        out
    }

    /// Emit the longest complete UTF-8 prefix of the buffer.
    fn drain_complete(&mut self) -> String {
        // walk back up to 3 bytes looking for a non-continuation lead
        let n = self.buf.len();
        let mut cut = n;
        for back in 1..=3.min(n) {
            let b = self.buf[n - back];
            if b & 0xC0 != 0x80 {
                // lead byte: is the sequence complete?
                let want = if b >= 0xF0 {
                    4
                } else if b >= 0xE0 {
                    3
                } else if b >= 0xC0 {
                    2
                } else {
                    1
                };
                if back < want {
                    cut = n - back; // incomplete sequence stays buffered
                }
                break;
            }
        }
        let out = String::from_utf8_lossy(&self.buf[..cut]).to_string();
        self.buf.drain(..cut);
        out
    }
}

/// GPT-2 byte→unicode table: printable bytes identity; others shifted to
/// 256+n in increasing byte order.
pub fn build_byte_encoder() -> HashMap<u8, char> {
    let keep = |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut m = HashMap::with_capacity(256);
    let mut n: u32 = 0;
    for b in 0..=255u8 {
        if keep(b) {
            m.insert(b, b as char);
        }
    }
    for b in 0..=255u8 {
        if !keep(b) {
            m.insert(b, char::from_u32(256 + n).expect("codepoint in range"));
            n += 1;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Tokenizer {
        // tiny hand-built vocab over "Hello"
        let vf = VocabFile {
            model: "gpt2".into(),
            pre: "qwen35".into(),
            tokens: ["H", "e", "l", "o", "He", "Hel", "lo", "Hello"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            token_types: vec![TYPE_NORMAL; 8],
            merges: vec!["H e".to_string(), "He l".to_string(), "l o".to_string(), "Hel lo".to_string()],
            eos: None,
            bos: None,
            pad: None,
            add_bos: false,
        };
        let mut id_by_token = HashMap::new();
        for (i, t) in vf.tokens.iter().enumerate() {
            id_by_token.insert(t.clone(), i as u32);
        }
        let mut ranks = HashMap::new();
        for (i, m) in vf.merges.iter().enumerate() {
            let (a, b) = m.split_once(' ').unwrap();
            ranks.insert((a.to_string(), b.to_string()), i as u32);
        }
        let byte_encoder = build_byte_encoder();
        let byte_decoder = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();
        Tokenizer::load_from_parts(
            vf,
            id_by_token,
            ranks,
            byte_encoder,
            byte_decoder,
            fancy_regex::Regex::new(QWEN35_PRE_PATTERN).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_byte_encoder_known_values() {
        let e = build_byte_encoder();
        assert_eq!(e[&b'A'], 'A');
        assert_eq!(e[&b' '], '\u{0120}'); // Ġ
        assert_eq!(e[&0x00], '\u{0100}'); // Ā
        assert_eq!(e[&b'\n'], '\u{010A}'); // 10 -> 256+10
        // roundtrip all 256
        let d: HashMap<char, u8> = e.iter().map(|(&b, &c)| (c, b)).collect();
        for b in 0..=255u8 {
            assert_eq!(d[&e[&b]], b);
        }
    }

    #[test]
    fn test_bpe_merges_to_expected_token() {
        let t = toy();
        // "Hello" -> H e l l o -> He l l o -> Hel l o -> Hel lo -> Hello
        assert_eq!(t.encode_piece("Hello").unwrap(), vec![7]);
    }

    #[test]
    fn test_bpe_single_symbol_fallback() {
        let t = toy();
        // "o" alone: single-char fast path
        assert_eq!(t.encode_piece("o").unwrap(), vec![3]);
    }

    #[test]
    fn test_pre_tokenizer_pieces() {
        let t = toy();
        let mut pieces = Vec::new();
        let text = "I'm a  student\n";
        for m in t.pre_regex.find_iter(text) {
            pieces.push(m.unwrap().as_str().to_string());
        }
        assert_eq!(
            pieces,
            vec!["I", "'m", " a", " ", " student", "\n"],
            "contraction alternation, lookahead split of double space"
        );
    }

    #[test]
    fn test_streaming_decoder_partial_utf8() {
        let t = toy();
        // 'é' = 0xC3 0xA9 — force it through two pushes via crafted vocab
        // entries is overkill; push raw bytes by splitting a decoded id is
        // not possible here — test the buffer logic directly instead.
        let mut d = StreamingDecoder::new(t);
        d.buf = vec![0xC3]; // incomplete 2-byte lead
        assert_eq!(d.drain_complete(), "", "incomplete sequence buffers");
        d.buf.push(0xA9);
        assert_eq!(d.drain_complete(), "é");
    }
}
