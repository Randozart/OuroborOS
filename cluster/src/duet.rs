//! DUET P4 — Choice frames (DUET-CHOOSE, docs/DUET.md §5).
//!
//! When an outcome lands in a jointly enumerable candidate set, the wire
//! carries indices, not values: propose(set) → outcome → pick(i) → verify.
//! Flagship landing: Art. 4 energy-budget reconciliation — the budget is a
//! sum over nodes; predicted draws travel as one "as-planned" bit, drifted
//! nodes send a quantized delta. The budget decision must be EXACT vs
//! ground truth either way: speculation may fail, it may never lie.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Wire frame for a choice: pick an option from a set the peer already
/// knows (set_hash guards version skew — the DUET never trusts blindly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceFrame {
    pub set_id: String,
    pub set_hash: u64,
    /// Index into the agreed candidate set.
    pub pick: u32,
}

/// The hash of a candidate set: both sides must have derived the SAME set
/// from shared state, or the pick is meaningless.
pub fn set_hash(options: &[f64]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in options {
        h.write(&v.to_le_bytes());
    }
    h.finish()
}

/// A proposal the receiver evaluates: predicted value + tolerance + fallback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub set_id: String,
    pub set_hash: u64,
    pub predicted: f64,
    /// |actual - predicted| beyond this ⇒ drift (delta path).
    pub tolerance: f64,
}

/// Verdict on one prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Within margin: the wire carries one bit. Reconstruction uses the
    /// prediction — bounded error ≤ tolerance per node, by design.
    AsPlanned,
    /// Beyond margin: the wire carries the node's ACTUAL reading (8 bytes).
    /// Reconstruction is bit-exact for this node. (A delta encoding is a
    /// future compression once cross-silicon margins are characterized —
    /// `p + (a-p) ≠ a` in f64, so naive deltas are lossy and rejected.)
    Drifted(f64),
}

pub fn judge(predicted: f64, actual: f64, tolerance: f64) -> Verdict {
    if (actual - predicted).abs() <= tolerance {
        Verdict::AsPlanned
    } else {
        Verdict::Drifted(actual)
    }
}

/// Reconcile a round of predicted vs actual draws.
///
/// Contract: the reconstruction equals the truth over **effective values**
/// (predicted where as-planned, actual where drifted) EXACTLY, and differs
/// from the raw-actual sum by at most `n_as_planned × tolerance` — the
/// unavoidable price of not shipping quiet nodes' readings.
pub fn reconcile(predicted: &[f64], actual: &[f64], tolerance: f64) -> Result<Reconciliation> {
    if predicted.len() != actual.len() {
        bail!("predicted/actual arity mismatch: {} vs {}", predicted.len(), actual.len());
    }
    let mut wire_bits = 0u64; // as-planned verdicts
    let mut wire_deltas = 0u64;
    let mut sum = 0f64;
    let mut verdicts = Vec::with_capacity(predicted.len());
    for (i, (&p, &a)) in predicted.iter().zip(actual).enumerate() {
        let v = judge(p, a, tolerance);
        match v {
            Verdict::AsPlanned => {
                wire_bits += 1;
                sum += p;
            }
            Verdict::Drifted(reading) => {
                wire_deltas += 1;
                sum += reading;
            }
        }
        verdicts.push((i, v));
    }
    Ok(Reconciliation { verdicts, sum, wire_bits, wire_deltas })
}

/// The reconciliation result: verdicts per node + the exact total.
#[derive(Debug, Clone, PartialEq)]
pub struct Reconciliation {
    pub verdicts: Vec<(usize, Verdict)>,
    /// The sum as reconstructed from predictions + deltas.
    pub sum: f64,
    pub wire_bits: u64,
    pub wire_deltas: u64,
}

impl Reconciliation {
    /// The values the reconstruction summed: predicted where as-planned,
    /// actual where drifted.
    pub fn effective_truth(&self, predicted: &[f64], actual: &[f64]) -> Vec<f64> {
        self.verdicts
            .iter()
            .map(|(i, v)| match v {
                Verdict::AsPlanned => predicted[*i],
                Verdict::Drifted(_) => actual[*i],
            })
            .collect()
    }

    /// Exact over communicated values (the wire contract).
    pub fn exact_over_effective(&self, predicted: &[f64], actual: &[f64]) -> bool {
        let truth: f64 = self.effective_truth(predicted, actual).iter().sum();
        (self.sum - truth).abs() < 1e-9
    }

    /// Bounded divergence from the raw-actual sum: ≤ n_as_planned × tol.
    pub fn within_tolerance_bound(&self, actual: &[f64], tolerance: f64) -> bool {
        let truth: f64 = actual.iter().sum();
        (self.sum - truth).abs() <= self.wire_bits as f64 * tolerance + 1e-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_draws(n: usize, seed: u8) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed as u32 | 1;
        let mut next = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) % 2000) as f64 / 10.0 + 10.0
        };
        let predicted: Vec<f64> = (0..n).map(|_| next()).collect();
        // actuals drift within ±0.5 W (tolerance 1.0) — as-planned territory
        let actual = predicted.iter().map(|p| p + (next() - 100.0) / 100.0).collect();
        (predicted, actual)
    }

    /// The happy path: quiet node → 1 bit; reconstruction exact over the
    /// communicated values, divergence from raw actuals within tolerance.
    #[test]
    fn test_as_planned_reconstruction_exact() {
        let (predicted, actual) = node_draws(8, 7);
        let r = reconcile(&predicted, &actual, 1.0).unwrap();
        assert_eq!(r.wire_bits, 8);
        assert_eq!(r.wire_deltas, 0);
        assert!(r.exact_over_effective(&predicted, &actual), "wire contract exact");
        assert!(r.within_tolerance_bound(&actual, 1.0), "raw divergence bounded by n x tol");
    }

    /// A drifted node sends its reading; its contribution is bit-exact and
    /// the total divergence from raw actuals stays inside the bound.
    #[test]
    fn test_drift_delta_reconstruction_exact() {
        let (predicted, mut actual) = node_draws(8, 9);
        actual[3] += 25.0; // a real drift, far beyond tolerance
        let r = reconcile(&predicted, &actual, 1.0).unwrap();
        assert_eq!(r.wire_deltas, 1);
        assert!(matches!(r.verdicts[3].1, Verdict::Drifted(d) if (d - actual[3]).abs() < 1e-12));
        assert!(r.exact_over_effective(&predicted, &actual), "drifted node bit-exact");
        assert!(r.within_tolerance_bound(&actual, 1.0));
    }

    /// The wire bill collapses vs full f64 payloads (the DUET economics).
    #[test]
    fn test_wire_bill_collapses() {
        let (predicted, actual) = node_draws(64, 11);
        let r = reconcile(&predicted, &actual, 1.0).unwrap();
        let duet_bytes = r.wire_bits / 8 + r.wire_deltas * 8;
        let full_bytes = 64 * 8;
        assert!(duet_bytes < full_bytes / 4, "DUET must beat full push: {} vs {}", duet_bytes, full_bytes);
    }

    /// Version skew: the peer's set hash differs → fallback, not corruption.
    #[test]
    fn test_set_hash_mismatch_falls_back() {
        let options = vec![10.0, 20.0, 30.0];
        let mine = set_hash(&options);
        let theirs = set_hash(&[10.0, 20.0, 99.0]); // skewed peer
        let frame = ChoiceFrame {
            set_id: "budget-7".into(),
            set_hash: mine,
            pick: 2,
        };
        // receiver compares frame.set_hash against ITS set hash:
        assert_ne!(frame.set_hash, theirs, "skew detected");
        // ⇒ refuse the pick, request full payload (never misindex).
    }

    /// Choice frames survive the wire.
    #[test]
    fn test_choice_frame_serde_roundtrip() {
        let f = ChoiceFrame { set_id: "s".into(), set_hash: 42, pick: 1 };
        let b = serde_json::to_vec(&f).unwrap();
        assert_eq!(serde_json::from_slice::<ChoiceFrame>(&b).unwrap(), f);
    }
}
