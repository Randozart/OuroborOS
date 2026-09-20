//! Bonsai text generation, local to the shell — `ask <text>.`
//!
//! The shell's head loads the n1 shards (mmap — instant) and runs greedy
//! generation with the attached tokenizer: text in, streamed text out.
//! Late-2026 reality on the Ivy Bridge head: ~4 s/token warm; `ask` is
//! honest about that and reports the stream as it arrives.

use anyhow::{Context, Result};
use ouro_cluster::infer::qwen35::{Card, Qwen35Model};
use ouro_cluster::infer::tokenizer::StreamingDecoder;

/// Run one greedy completion. Returns (completion_text, tokens_emitted).
pub fn ask(dir: &str, text: &str, max_tokens: usize, mut emit: impl FnMut(&str)) -> Result<String> {
    let shard = format!("{dir}/shard_1.bmts");
    if !std::path::Path::new(&format!("{dir}/tokenizer.json")).exists() {
        anyhow::bail!(
            "no tokenizer at {dir} — run: python3 tools/dump_tokenizer.py <gguf> {dir}"
        );
    }
    let mut model = Qwen35Model::load(&[shard.as_str()], Card::load_dir(dir)?)
        .context("loading bonsai shards")?;
    let tok = model
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("no tokenizer attached"))?
        .clone();
    let _ = &tok;

    let prompt_ids = model.tokenize(text)?;
    let mut streamer = StreamingDecoder::new(
        ouro_cluster::infer::tokenizer::Tokenizer::load_dir(dir)?,
    );
    let eos = tok.eos().unwrap_or(u32::MAX) as usize;

    let mut tok_id = *prompt_ids.last().context("empty prompt")? as usize;
    let mut emitted = 0usize;
    let mut full = String::new();
    for _ in 0..max_tokens {
        let h = model.step(tok_id)?;
        let l = model.logits(&h)?;
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
        emitted += 1;
    }
    full.push_str(&streamer.finish());
    let _ = emitted;
    Ok(full)
}
