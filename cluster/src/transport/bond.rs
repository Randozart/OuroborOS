//! The bond layer — a scheduler over priced lanes (docs/AIR_PATH.md §4.2).
//!
//! A peer has N live lanes (one per `PricedEdge`). Frames ride whichever
//! lane(s) the policy picks: control on the lowest-latency lane, bulk
//! striped across all lanes, critical duplicated on two. The policy is a
//! pure function of the priced edge set — no sockets here. The transport
//! rung opens one channel per lane and applies the picks; frames already
//! carry seq numbers, so cross-path reordering is already handled.

use serde::{Deserialize, Serialize};

use super::auth::Secret;
use super::edge::PricedEdge;
use super::frames::{pump_send, FrameSession, DEFAULT_CHUNK, DEFAULT_WINDOW};
use anyhow::Result;
use std::io::Read;
use std::net::TcpStream;

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

/// One authenticated frame channel on a lane (docs/AIR_PATH.md §4.2).
pub struct BondChannel {
    pub edge: PricedEdge,
    pub session: FrameSession<TcpStream>,
}

/// The bond runtime: N frame channels to a peer, one per live lane. Frames
/// ride whichever lane(s) `schedule` picks per class — the doctrine's "both
/// paths at once" as code. Each channel is an independent `FrameSession`;
/// seq numbers already order cross-path delivery (frames.rs).
pub struct Bond {
    channels: Vec<BondChannel>,
}

impl Bond {
    pub fn new() -> Self {
        Self { channels: Vec::new() }
    }

    /// Open one authenticated frame channel for `edge` to `addr`. In a real
    /// multi-homed deployment each lane is its own IP; the caller maps
    /// edge -> address here (the address table is a follow-on rung).
    pub fn connect(&mut self, edge: PricedEdge, secret: &Secret, addr: &str) -> Result<()> {
        let stream = TcpStream::connect(addr)?;
        self.channels.push(BondChannel {
            edge,
            session: FrameSession::new(stream, *secret),
        });
        Ok(())
    }

    pub fn lane_count(&self) -> usize {
        self.channels.len()
    }

    /// The priced edge set, in channel order.
    pub fn edges(&self) -> Vec<&PricedEdge> {
        self.channels.iter().map(|c| &c.edge).collect()
    }

    /// Stream `src` to the peer over the lane `schedule` picks for `class`
    /// (bulk stripes by `round`). Returns (lane iface, bytes sent).
    pub fn send(&mut self, class: FrameClass, src: &mut impl Read, round: usize) -> Result<(String, u64)> {
        let edge_set: Vec<PricedEdge> = self.channels.iter().map(|c| c.edge.clone()).collect();
        let choice = schedule(class, &edge_set, round);
        let iface = choice
            .primary
            .ok_or_else(|| anyhow::anyhow!("no lane available"))?;
        let channel = self
            .channels
            .iter_mut()
            .find(|c| c.edge.iface == iface)
            .ok_or_else(|| anyhow::anyhow!("lane {} not open", iface))?;
        let sent = pump_send(src, &mut channel.session, DEFAULT_CHUNK, DEFAULT_WINDOW)?;
        Ok((iface, sent))
    }
}

impl Default for Bond {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::edge::EdgeKind;
    use crate::transport::frames::pump_recv;

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

    const KEY: Secret = [7u8; 32];

    /// The bond runtime over real loopback TCP: two frame channels (two
    /// "lanes") to one peer, `send` routes a payload over the lane the
    /// policy picks, and the bytes arrive intact on that lane. The doctrine's
    /// "both paths at once" — a fetch rides the chosen lane(s).
    #[test]
    fn test_bond_sends_over_chosen_lane() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let data: Vec<u8> = (0..400_000u32).map(|i| (i % 251) as u8).collect();

        // Server: accept one connection per lane, pump each into its own sink.
        let server = std::thread::spawn(move || {
            let (s1, _) = listener.accept().unwrap();
            let (s2, _) = listener.accept().unwrap();
            let mut sess1 = FrameSession::new(s1, KEY);
            let mut sess2 = FrameSession::new(s2, KEY);
            let mut dst1 = Vec::new();
            let mut dst2 = Vec::new();
            let (n1, _) = pump_recv(&mut sess1, &mut dst1, DEFAULT_WINDOW).unwrap();
            let (n2, _) = pump_recv(&mut sess2, &mut dst2, DEFAULT_WINDOW).unwrap();
            (n1, dst1, n2, dst2)
        });

        let mut bond = Bond::new();
        let primary = lane("enp3s0", EdgeKind::Tcp, 1000, 300, 10);
        let air = lane("wlan0", EdgeKind::Air, 60, 2000, 500);
        bond.connect(primary.clone(), &KEY, &addr.to_string()).unwrap();
        bond.connect(air.clone(), &KEY, &addr.to_string()).unwrap();
        assert_eq!(bond.lane_count(), 2);

        // Round 0: bulk takes the fastest lane (copper, 1000 Mbit/s).
        let mut src = std::io::Cursor::new(data.clone());
        let (iface, sent) = bond.send(FrameClass::Bulk, &mut src, 0).unwrap();
        assert_eq!(iface, "enp3s0");
        assert_eq!(sent as usize, data.len());

        // Round 1: stripe to the next lane (air).
        let mut src2 = std::io::Cursor::new(data.clone());
        let (iface2, _) = bond.send(FrameClass::Bulk, &mut src2, 1).unwrap();
        assert_eq!(iface2, "wlan0");

        let (n1, dst1, n2, dst2) = server.join().unwrap();
        assert_eq!(n1 as usize, data.len());
        assert_eq!(n2 as usize, data.len());
        assert_eq!(dst1, data, "primary lane delivered intact");
        assert_eq!(dst2, data, "striped lane delivered intact");
    }

    /// `Bond::send` with no lanes refuses, never panics.
    #[test]
    fn test_bond_no_lanes_refuses() {
        let mut bond = Bond::new();
        let mut src = std::io::Cursor::new(vec![0u8; 8]);
        let err = bond.send(FrameClass::Bulk, &mut src, 0);
        assert!(err.is_err());
    }
}