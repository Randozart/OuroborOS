//! The bond layer — a scheduler over priced lanes (docs/AIR_PATH.md §4.2).
//!
//! A peer has N live lanes (one per `PricedEdge`). Frames ride whichever
//! lane(s) the policy picks: control on the lowest-latency lane, bulk
//! striped across all lanes, critical duplicated on two. The policy is a
//! pure function of the priced edge set — no sockets here. The transport
//! rung opens one channel per lane and applies the picks; frames already
//! carry seq numbers, so cross-path reordering is already handled.

use serde::{Deserialize, Serialize};

use crate::transport::edge::PricedEdge;

/// What a frame is for — the lane policy keys on it (docs/AIR_PATH.md §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameClass {
    /// Tiny latency-bound ops (attach/resolve/stat/ctl) — lowest-latency lane.
    Control,
    /// Large payloads (shards, checkpoints, prefill batches) — stripe.
    Bulk,
    /// State snapshots / rebind acks — duplicate for survivability.
    Critical,
}

/// Which lane(s) a frame of a given class rides. Both names refer to a
/// `PricedEdge.iface`; `primary` is always set when a lane exists.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LaneChoice {
    pub primary: Option<String>,
    pub secondary: Option<String>,
}

impl LaneChoice {
    /// No lane available (peer has no live edges).
    pub fn is_available(&self) -> bool {
        self.primary.is_some()
    }
}

/// Combined latency cost: RTT plus jitter. Air's larger jitter is exactly
/// why it loses control frames to copper when the base latencies are close.
fn lane_cost(e: &PricedEdge) -> u32 {
    e.latency_us.saturating_add(e.jitter_us)
}

/// Pick the lane(s) for a frame class over a peer's priced edge set.
///
/// - `Control`: the lowest-cost (latency + jitter) lane.
/// - `Bulk`: bandwidth-descending lanes, round-robin by `stripe_round` —
///   consecutive bulk frames fan out across every live lane.
/// - `Critical`: the two lowest-cost distinct lanes (duplicated frames).
pub fn schedule(class: FrameClass, edges: &[PricedEdge], stripe_round: usize) -> LaneChoice {
    if edges.is_empty() {
        return LaneChoice::default();
    }
    match class {
        FrameClass::Control => LaneChoice {
            primary: cheapest(edges).map(|e| e.iface.clone()),
            secondary: None,
        },
        FrameClass::Critical => {
            let mut ordered: Vec<&PricedEdge> = edges.iter().collect();
            ordered.sort_by_key(|e| lane_cost(e));
            let primary = ordered.first().map(|e| e.iface.clone());
            let secondary = ordered.get(1).map(|e| e.iface.clone());
            LaneChoice { primary, secondary }
        }
        FrameClass::Bulk => {
            let mut ordered: Vec<&PricedEdge> = edges.iter().collect();
            ordered.sort_by(|a, b| b.bw_mbps.cmp(&a.bw_mbps));
            let chosen = ordered[stripe_round % ordered.len()];
            LaneChoice {
                primary: Some(chosen.iface.clone()),
                secondary: None,
            }
        }
    }
}

/// The lowest-cost (latency + jitter) lane.
fn cheapest(edges: &[PricedEdge]) -> Option<&PricedEdge> {
    edges.iter().min_by_key(|e| lane_cost(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::edge::EdgeKind;

    fn lane(iface: &str, kind: EdgeKind, bw: u32, lat: u32, jit: u32) -> PricedEdge {
        PricedEdge {
            iface: iface.into(),
            kind,
            bw_mbps: bw,
            latency_us: lat,
            jitter_us: jit,
            watts: 1,
            signal_dbm: 0,
        }
    }

    fn edges() -> Vec<PricedEdge> {
        vec![
            lane("enp3s0", EdgeKind::Tcp, 1000, 300, 10),
            lane("wlan0", EdgeKind::Air, 60, 2000, 500),
            lane("rdma0", EdgeKind::Rdma, 10000, 5, 2),
        ]
    }

    #[test]
    fn test_control_picks_lowest_latency() {
        let choice = schedule(FrameClass::Control, &edges(), 0);
        assert_eq!(choice.primary.as_deref(), Some("rdma0"));
        assert_eq!(choice.secondary, None);
    }

    #[test]
    fn test_bulk_stripes_bandwidth_desc() {
        // sorted by bw desc: rdma0, enp3s0, wlan0
        let e = edges();
        let c0 = schedule(FrameClass::Bulk, &e, 0);
        let c1 = schedule(FrameClass::Bulk, &e, 1);
        let c2 = schedule(FrameClass::Bulk, &e, 2);
        let c3 = schedule(FrameClass::Bulk, &e, 3);
        assert_eq!(c0.primary.as_deref(), Some("rdma0"));
        assert_eq!(c1.primary.as_deref(), Some("enp3s0"));
        assert_eq!(c2.primary.as_deref(), Some("wlan0"));
        assert_eq!(c3.primary.as_deref(), Some("rdma0"), "round-robin wraps");
    }

    #[test]
    fn test_critical_duplicates_two_lowest_cost() {
        let choice = schedule(FrameClass::Critical, &edges(), 0);
        assert_eq!(choice.primary.as_deref(), Some("rdma0"));
        assert_eq!(choice.secondary.as_deref(), Some("enp3s0"));
    }

    #[test]
    fn test_empty_edges_no_choice() {
        let choice = schedule(FrameClass::Control, &[], 0);
        assert!(!choice.is_available());
        let choice = schedule(FrameClass::Bulk, &[], 5);
        assert!(!choice.is_available());
        let choice = schedule(FrameClass::Critical, &[], 0);
        assert_eq!(choice.primary, None);
        assert_eq!(choice.secondary, None);
    }

    #[test]
    fn test_control_prefers_copper_when_latency_close() {
        // air with tiny latency but large jitter must lose to copper.
        let e = vec![
            lane("enp3s0", EdgeKind::Tcp, 1000, 100, 5),
            lane("wlan0", EdgeKind::Air, 60, 90, 200),
        ];
        let choice = schedule(FrameClass::Control, &e, 0);
        assert_eq!(choice.primary.as_deref(), Some("enp3s0"));
    }

    #[test]
    fn test_single_lane_always_selected() {
        let e = vec![lane("wlan0", EdgeKind::Air, 60, 2000, 500)];
        assert_eq!(schedule(FrameClass::Control, &e, 0).primary.as_deref(), Some("wlan0"));
        assert_eq!(schedule(FrameClass::Bulk, &e, 0).primary.as_deref(), Some("wlan0"));
        let c = schedule(FrameClass::Critical, &e, 0);
        assert_eq!(c.primary.as_deref(), Some("wlan0"));
        assert_eq!(c.secondary, None, "one lane cannot duplicate");
    }
}