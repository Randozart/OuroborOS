pub mod beast;

/// Cap'n Proto metadata schemas (BMTS v2) — generated at build time from
/// schemas/*.capnp; requires the `capnp2` feature (and the capnp tool).
#[cfg(feature = "capnp2")]
pub mod bmts_capnp {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/bmts_capnp.rs"));
}
#[cfg(feature = "capnp2")]
pub mod model_capnp {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/model_capnp.rs"));
}

pub mod bmts;
pub mod duet;
pub mod pipeline;
pub mod infer;
pub mod error;
pub mod error_recovery;
pub mod op;
pub mod probe;
pub mod registry;
pub mod scheduler;
pub mod sync;
pub mod transport;
pub mod update;
pub mod weights;
