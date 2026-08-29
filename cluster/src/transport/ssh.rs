//! SSH-based transport (MVP).
//!
//! Dispatches tasks to nodes via SSH. ~1ms overhead per task.
//! Phase 2 replaces this with a custom TCP protocol.
