use serde::{Deserialize, Serialize};

/// The typed lane registry (docs/AIR_PATH.md §1.1, §4.1). Every physical
/// avenue that can carry a connection is a lane; "protocols be damned".
/// `Other` keeps the door open for an avenue with no kind yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Signed line + frames over TCP (transport/auth.rs, frames.rs).
    Tcp,
    /// Raw L2 Ethernet, EtherType 0x88B5, magic 0x4F55524F.
    L2,
    /// PCIe bus-master (transport/dma.rs, ouro-dma).
    Dma,
    /// InfiniBand / RoCE (SoftRoCE rxe0).
    Rdma,
    /// BMTS raw block device / storage lane.
    Blk,
    /// NVMe / SATA (checkpoint plane).
    Nvme,
    /// USB / Thunderbolt.
    Usb,
    /// Powerline (Ethernet over AC).
    Pwr,
    /// Serial / UART / any pin.
    Serial,
    /// GPU interconnect (NVLink/SLI).
    Gpu,
    /// Display port / HDMI — an out-port is an avenue (Art. 5).
    Display,
    /// Wi-Fi.
    Air,
    /// Radio / modem.
    Modem,
    /// Bluetooth / BLE.
    Bt,
    /// Cellular (the internet join edge, Art. 10 mTLS).
    Cellular,
    /// LoRa / ISM band.
    Lora,
    /// IR / light (Li-Fi).
    Optical,
    /// Acoustic (sound card as modem).
    Acoustic,
    /// An avenue with no kind yet (protocols be damned).
    Other(String),
}

impl EdgeKind {
    /// A short human label for rendering.
    pub fn label(&self) -> String {
        match self {
            EdgeKind::Tcp => "tcp".into(),
            EdgeKind::L2 => "l2".into(),
            EdgeKind::Dma => "dma".into(),
            EdgeKind::Rdma => "rdma".into(),
            EdgeKind::Blk => "blk".into(),
            EdgeKind::Nvme => "nvme".into(),
            EdgeKind::Usb => "usb".into(),
            EdgeKind::Pwr => "pwr".into(),
            EdgeKind::Serial => "serial".into(),
            EdgeKind::Gpu => "gpu".into(),
            EdgeKind::Display => "display".into(),
            EdgeKind::Air => "air".into(),
            EdgeKind::Modem => "modem".into(),
            EdgeKind::Bt => "bt".into(),
            EdgeKind::Cellular => "cell".into(),
            EdgeKind::Lora => "lora".into(),
            EdgeKind::Optical => "optical".into(),
            EdgeKind::Acoustic => "acoustic".into(),
            EdgeKind::Other(s) => format!("other({})", s),
        }
    }
}

/// A priced lane between two nodes. Measured, never spec-sheet
/// (docs/AIR_PATH.md §4.1); the scheduler sees only the price tuple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricedEdge {
    /// Interface name: enp3s0 | wlan0 | rdma0 | hdmi0 | ...
    pub iface: String,
    pub kind: EdgeKind,
    /// Measured bandwidth in Mbit/s (link speed or `iw` tx bitrate).
    pub bw_mbps: u32,
    /// Measured round-trip in microseconds against the head.
    pub latency_us: u32,
    /// Measured jitter (ping mdev) in microseconds — air lives and dies on this.
    pub jitter_us: u32,
    /// Active draw estimate in watts (Art. 4); powercap measurement is a
    /// Phase C refinement.
    pub watts: u32,
    /// Wi-Fi signal in dBm (negative); 0 when not an air lane.
    pub signal_dbm: i32,
}

impl PricedEdge {
    /// True when this lane is an air/radio lane.
    pub fn is_air(&self) -> bool {
        matches!(self.kind, EdgeKind::Air | EdgeKind::Modem | EdgeKind::Bt | EdgeKind::Cellular | EdgeKind::Lora)
    }

    /// Human-readable one-liner.
    pub fn describe(&self) -> String {
        format!(
            "{} ({}, {} Mbit/s, {}us+/-{}us, {}W{})",
            self.iface,
            self.kind.label(),
            self.bw_mbps,
            self.latency_us,
            self.jitter_us,
            self.watts,
            if self.signal_dbm != 0 { format!(", {} dBm", self.signal_dbm) } else { String::new() },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_kind_serde_round_trip() {
        for kind in [
            EdgeKind::Tcp,
            EdgeKind::L2,
            EdgeKind::Air,
            EdgeKind::Display,
            EdgeKind::Other("acoustic-coupler".into()),
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: EdgeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn test_edge_labels() {
        assert_eq!(EdgeKind::Air.label(), "air");
        assert_eq!(EdgeKind::Display.label(), "display");
        assert_eq!(EdgeKind::Other("x".into()).label(), "other(x)");
    }

    #[test]
    fn test_air_detection() {
        assert!(PricedEdge {
            iface: "wlan0".into(),
            kind: EdgeKind::Air,
            bw_mbps: 60,
            latency_us: 2000,
            jitter_us: 500,
            watts: 2,
            signal_dbm: -45,
        }
        .is_air());
        assert!(!PricedEdge {
            iface: "enp3s0".into(),
            kind: EdgeKind::Tcp,
            bw_mbps: 1000,
            latency_us: 300,
            jitter_us: 10,
            watts: 1,
            signal_dbm: 0,
        }
        .is_air());
    }

    #[test]
    fn test_describe() {
        let e = PricedEdge {
            iface: "wlan0".into(),
            kind: EdgeKind::Air,
            bw_mbps: 60,
            latency_us: 2000,
            jitter_us: 500,
            watts: 2,
            signal_dbm: -45,
        };
        assert_eq!(
            e.describe(),
            "wlan0 (air, 60 Mbit/s, 2000us+/-500us, 2W, -45 dBm)"
        );
    }
}