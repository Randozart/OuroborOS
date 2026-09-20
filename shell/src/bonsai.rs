//! Bonsai text generation, local to the shell — `ask <text>.`
//!
//! The shell's head loads the n1 shards (mmap — instant) and runs greedy
//! generation with the attached tokenizer: text in, streamed text out.
//! The session is PERSISTENT across REPL turns: prompt tokens are ingested
//! into the recurrent state, so follow-up `ask` turns continue the
//! conversation. `ask clear` drops the context (full re-ingest on next ask).
//!
//! Late-2026 reality on the Ivy Bridge head: ~4 s/token warm; `ask` is
//! honest about that and streams as it goes.

use anyhow::{bail, Context, Result};
use ouro_cluster::infer::qwen35::{Card, Qwen35Model};
use ouro_cluster::infer::tokenizer::{StreamingDecoder, Tokenizer};

/// A persistent (multi-turn) Bonsai completion session.
pub struct BonsaiSession {
    model: Qwen35Model,
    tok: Tokenizer,
    scratch: Tokenizer,
}

impl BonsaiSession {
    pub fn open(dir: &str) -> Result<Self> {
        if !std::path::Path::new(&format!("{dir}/tokenizer.json")).exists() {
            bail!("no tokenizer at {dir} — run: python3 tools/dump_tokenizer.py <gguf> {dir}");
        }
        let shard = format!("{dir}/shard_1.bmts");
        let model = Qwen35Model::load(&[shard.as_str()], Card::load_dir(dir)?)
            .context("loading bonsai shards")?;
        let tok = Tokenizer::load_dir(dir)?;
        let scratch = tok.clone();
        Ok(Self { model, tok, scratch })
    }

    /// Drop conversation context (recurrent state + KV + position).
    pub fn clear(&mut self) {
        self.model.reset();
    }

    /// Greedy continuation of `text`, ingesting the prompt into the
    /// persistent context first. Returns the completion text.
    pub fn ask(&mut self, text: &str, max_tokens: usize, mut emit: impl FnMut(&str)) -> Result<String> {
        let prompt_ids = self.model.tokenize(text)?;
        if prompt_ids.is_empty() {
            bail!("empty prompt");
        }
        // ingest all prompt tokens but the last into the recurrent state;
        // the generation loop steps the last one itself
        for &t in &prompt_ids[..prompt_ids.len() - 1] {
            self.model.step(t as usize)?;
        }
        let mut tok_id = *prompt_ids.last().unwrap() as usize;
        let eos = self.tok.eos().unwrap_or(u32::MAX) as usize;

        let mut streamer = StreamingDecoder::new(self.scratch.clone());
        let mut full = String::new();
        for _ in 0..max_tokens {
            let h = self.model.step(tok_id)?;
            let l = self.model.logits(&h)?;
            let next = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if next == eos {
                emit("<eos>");
                break;
            }
            let piece = streamer.push(next as u32);
            emit(&piece);
            full.push_str(&piece);
            tok_id = next;
        }
        full.push_str(&streamer.finish());
        Ok(full)
    }

}

