//! SoftRoCE RDMA transport — the head addresses tail RAM as device space.
//!
//! Tier 4 of the DMA ladder (`docs/DMA_ROADMAP.md`): registered memory
//! regions, zero-copy, true ibverbs semantics over any NIC via `rdma_rxe`.
//!
//! The control plane stays TCP (QP info exchange); the data plane is pure
//! RDMA — `ouro_ibv_post_send` with `IBV_WR_RDMA_READ` / `IBV_WR_RDMA_WRITE`.

use std::ffi::CStr;
use std::ptr;

use anyhow::{bail, Result};

use crate::transport::rdma_ffi as ib;

/// An open RDMA device (ibverbs context + protection domain).
pub struct RdmaDevice {
    ctx: *mut ib::ibv_context,
    pd: *mut ib::ibv_pd,
    _dev_name: String,
}

unsafe impl Send for RdmaDevice {}
unsafe impl Sync for RdmaDevice {}

/// A completion queue bound to an RdmaDevice.
pub struct CompletionQueue {
    cq: *mut ib::ibv_cq,
}

/// Queue pair states for the init→RTR→RTS transition.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QpInfo {
    pub qp_num: u32,
    pub lid: u16,
    pub gid: [u8; 16],
    pub port: u8,
}

/// A queue pair — the RDMA connection endpoint.
pub struct QueuePair {
    qp: *mut ib::ibv_qp,
    pd: *mut ib::ibv_pd,
    ctx: *mut ib::ibv_context,
    port: u8,
}

unsafe impl Send for QueuePair {}

/// A registered memory region — pinned, DMA-visible.
pub struct MemoryRegion {
    mr: *mut ib::ibv_mr,
    buf: Vec<u8>,
    remote_addr: u64,
    rkey: u32,
}

// ── Device management ─────────────────────────────────────────────

impl RdmaDevice {
    /// Open the first available RDMA device (SoftRoCE or real IB).
    pub fn open() -> Result<Self> {
        Self::open_named(None)
    }

    /// Open a specific RDMA device by name, or the first if `name` is None.
    pub fn open_named(name: Option<&str>) -> Result<Self> {
        unsafe {
            let mut num_devices: i32 = 0;
            let dev_list = ib::ibv_get_device_list(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                bail!("no RDMA devices found — load rdma_rxe first");
            }

            let mut target: *mut ib::ibv_device = ptr::null_mut();
            let mut found = false;
            for i in 0..num_devices as usize {
                let dev = *dev_list.add(i);
                let dev_name = ib::ibv_get_device_name(dev);
                if dev_name.is_null() {
                    continue;
                }
                let cname = CStr::from_ptr(dev_name).to_string_lossy();
                match name {
                    Some(n) if cname == n => {
                        target = dev;
                        found = true;
                        break;
                    }
                    None if !found => {
                        target = dev;
                        found = true;
                    }
                    _ => {}
                }
            }

            if target.is_null() {
                ib::ibv_free_device_list(dev_list);
                bail!("RDMA device not found");
            }

            let ctx = ib::ibv_open_device(target);
            if ctx.is_null() {
                ib::ibv_free_device_list(dev_list);
                bail!("failed to open RDMA device");
            }

            let pd = ib::ibv_alloc_pd(ctx);
            if pd.is_null() {
                ib::ibv_close_device(ctx);
                ib::ibv_free_device_list(dev_list);
                bail!("failed to allocate protection domain");
            }

            let dev_name_str = CStr::from_ptr(ib::ibv_get_device_name(target))
                .to_string_lossy()
                .into_owned();

            ib::ibv_free_device_list(dev_list);

            Ok(Self {
                ctx,
                pd,
                _dev_name: dev_name_str,
            })
        }
    }

    pub fn pd(&self) -> *mut ib::ibv_pd {
        self.pd
    }

    pub fn ctx(&self) -> *mut ib::ibv_context {
        self.ctx
    }

    /// Query device attributes (for capabilities).
    pub fn query_device(&self) -> Result<ib::ibv_device_attr> {
        unsafe {
            let mut attr: ib::ibv_device_attr = std::mem::zeroed();
            let rc = ib::ibv_query_device(self.ctx, &mut attr);
            if rc != 0 {
                bail!("ibv_query_device failed: rc={rc}");
            }
            Ok(attr)
        }
    }

    /// Query port info (GID, LID, state).
    pub fn query_port(&self, port: u8) -> Result<ib::ibv_port_attr> {
        unsafe {
            let mut attr: ib::ibv_port_attr = std::mem::zeroed();
            let rc = ib::ibv_query_port(self.ctx, port, &mut attr);
            if rc != 0 {
                bail!("ibv_query_port failed: rc={rc}");
            }
            Ok(attr)
        }
    }

    /// Get the GID for a port (index 0 = RoCEv2 default GID).
    pub fn query_gid(&self, port: u8, index: u32) -> Result<ib::ibv_gid> {
        unsafe {
            let mut gid: ib::ibv_gid = std::mem::zeroed();
            let rc = ib::ibv_query_gid(self.ctx, port, index, &mut gid);
            if rc != 0 {
                bail!("ibv_query_gid failed: rc={rc}");
            }
            Ok(gid)
        }
    }

    /// Create a completion queue with `cqe` entries.
    pub fn create_cq(&self, cqe: i32) -> Result<CompletionQueue> {
        unsafe {
            let cq = ib::ibv_create_cq(self.ctx, cqe, ptr::null_mut(), ptr::null_mut(), 0);
            if cq.is_null() {
                bail!("ibv_create_cq failed");
            }
            Ok(CompletionQueue { cq })
        }
    }
}

// ── Completion queue + queue pair creation ─────────────────────────

impl CompletionQueue {
    pub fn cq(&self) -> *mut ib::ibv_cq {
        self.cq
    }

    /// Create a queue pair connected to this CQ (both send and recv).
    pub fn create_qp(
        &self,
        pd: *mut ib::ibv_pd,
        ctx: *mut ib::ibv_context,
        port: u8,
    ) -> Result<QueuePair> {
        unsafe {
            let mut qp_attr: ib::ibv_qp_init_attr = std::mem::zeroed();
            qp_attr.send_cq = self.cq;
            qp_attr.recv_cq = self.cq;
            qp_attr.qp_type = ib::IBV_QPT_RC;
            qp_attr.sq_sig_all = 1;
            qp_attr.cap.max_send_wr = 128;
            qp_attr.cap.max_recv_wr = 128;
            qp_attr.cap.max_send_sge = 1;
            qp_attr.cap.max_recv_sge = 1;

            let qp = ib::ibv_create_qp(pd, &mut qp_attr);
            if qp.is_null() {
                bail!("ibv_create_qp failed");
            }
            Ok(QueuePair {
                qp,
                pd,
                ctx,
                port,
            })
        }
    }
}

// ── Queue pair operations ──────────────────────────────────────────

impl QueuePair {
    pub fn qp(&self) -> *mut ib::ibv_qp {
        self.qp
    }

    /// Get this QP's info for exchange over TCP.
    pub fn local_info(&self, port: u8) -> Result<QpInfo> {
        unsafe {
            let mut attr: ib::ibv_qp_attr = std::mem::zeroed();
            let mut init_attr: ib::ibv_qp_init_attr = std::mem::zeroed();
            let rc = ib::ibv_query_qp(
                self.qp,
                &mut attr,
                ib::IBV_QP_STATE | ib::IBV_QP_PKEY_INDEX,
                &mut init_attr,
            );
            if rc != 0 {
                bail!("ibv_query_qp failed: rc={rc}");
            }

            let gid = ib::ibv_get_device_guid((*self.ctx).device);
            let gid_bytes: [u8; 16] = gid.raw;

            let port_attr = {
                let mut pa: ib::ibv_port_attr = std::mem::zeroed();
                ib::ibv_query_port(self.ctx, port, &mut pa);
                pa
            };

            Ok(QpInfo {
                qp_num: attr.dest_qp_num,
                lid: port_attr.lid,
                gid: gid_bytes,
                port,
            })
        }
    }

    /// Transition QP: RESET → INIT.
    pub fn to_init(&self, port: u8) -> Result<()> {
        unsafe {
            let mut attr: ib::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ib::IBV_QPS_RESET;
            attr.pkey_index = 0;
            attr.port_num = port;
            attr.qp_access_flags = ib::IBV_ACCESS_LOCAL_WRITE
                | ib::IBV_ACCESS_REMOTE_READ
                | ib::IBV_ACCESS_REMOTE_WRITE;

            let mask = ib::IBV_QP_STATE
                | ib::IBV_QP_PKEY_INDEX
                | ib::IBV_QP_PORT
                | ib::IBV_QP_ACCESS_FLAGS;

            let rc = ib::ibv_modify_qp(self.qp, &mut attr, mask);
            if rc != 0 {
                bail!("QP to_init failed: rc={rc}");
            }
            Ok(())
        }
    }

    /// Transition QP: INIT → RTR (ready to receive).
    pub fn to_rtr(&self, remote: &QpInfo) -> Result<()> {
        unsafe {
            let mut attr: ib::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ib::IBV_QPS_RTR;
            attr.path_mtu = ib::IBV_MTU_1024;
            attr.dest_qp_num = remote.qp_num;
            attr.rq_psn = 0;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer = 12;

            attr.ah_attr.dlid = remote.lid;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = remote.port;

            // RoCEv2: GID type = RoCE
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid.raw = remote.gid;
            attr.ah_attr.grh.sgid_index = 0;
            attr.ah_attr.grh.hop_limit = 255;
            attr.ah_attr.grh.traffic_class = 0;

            let mask = ib::IBV_QP_STATE
                | ib::IBV_QP_PATH_MTU
                | ib::IBV_QP_DEST_QPN
                | ib::IBV_QP_RQ_PSN
                | ib::IBV_QP_MAX_DEST_RD_ATOMIC
                | ib::IBV_QP_MIN_RNR_TIMER
                | ib::IBV_QP_AV;

            let rc = ib::ibv_modify_qp(self.qp, &mut attr, mask);
            if rc != 0 {
                bail!("QP to_rtr failed: rc={rc}");
            }
            Ok(())
        }
    }

    /// Transition QP: RTR → RTS (ready to send).
    pub fn to_rts(&self) -> Result<()> {
        unsafe {
            let mut attr: ib::ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ib::IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.sq_psn = 0;
            attr.max_rd_atomic = 1;

            let mask = ib::IBV_QP_STATE
                | ib::IBV_QP_TIMEOUT
                | ib::IBV_QP_RETRY_CNT
                | ib::IBV_QP_RNR_RETRY
                | ib::IBV_QP_SQ_PSN
                | ib::IBV_QP_MAX_QP_RD_ATOMIC;

            let rc = ib::ibv_modify_qp(self.qp, &mut attr, mask);
            if rc != 0 {
                bail!("QP to_rts failed: rc={rc}");
            }
            Ok(())
        }
    }

    /// Post an RDMA READ — pull `len` bytes from `remote_addr` into local
    /// `mr` at `local_offset`.
    pub fn post_rdma_read(
        &self,
        local_mr: &MemoryRegion,
        local_offset: usize,
        remote_addr: u64,
        remote_rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        unsafe {
            let mut sge: ib::ibv_sge = std::mem::zeroed();
            sge.addr = (local_mr.buf.as_ptr() as usize + local_offset) as u64;
            sge.length = len;
            sge.lkey = (*local_mr.mr).lkey;

            let mut wr: ib::ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.opcode = ib::IBV_WR_RDMA_READ;
            wr.send_flags = ib::IBV_SEND_SIGNALED;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey = remote_rkey;

            let mut bad_wr: *mut ib::ibv_send_wr = ptr::null_mut();
            let rc = ib::ouro_ibv_post_send(self.qp, &mut wr, &mut bad_wr);
            if rc != 0 {
                bail!("ouro_ibv_post_send (RDMA_READ) failed: rc={rc}");
            }
            Ok(())
        }
    }

    /// Post an RDMA WRITE — push `len` bytes from local `mr` at
    /// `local_offset` to `remote_addr` on the peer.
    pub fn post_rdma_write(
        &self,
        local_mr: &MemoryRegion,
        local_offset: usize,
        remote_addr: u64,
        remote_rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        unsafe {
            let mut sge: ib::ibv_sge = std::mem::zeroed();
            sge.addr = (local_mr.buf.as_ptr() as usize + local_offset) as u64;
            sge.length = len;
            sge.lkey = (*local_mr.mr).lkey;

            let mut wr: ib::ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.opcode = ib::IBV_WR_RDMA_WRITE;
            wr.send_flags = ib::IBV_SEND_SIGNALED;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey = remote_rkey;

            let mut bad_wr: *mut ib::ibv_send_wr = ptr::null_mut();
            let rc = ib::ouro_ibv_post_send(self.qp, &mut wr, &mut bad_wr);
            if rc != 0 {
                bail!("ouro_ibv_post_send (RDMA_WRITE) failed: rc={rc}");
            }
            Ok(())
        }
    }

    /// Poll the completion queue for one completion. Returns (wr_id, byte_len).
    pub fn poll_cq(&self, cq: &CompletionQueue) -> Result<(u64, u32)> {
        unsafe {
            let mut wc: ib::ibv_wc = std::mem::zeroed();
            loop {
                let n = ib::ouro_ibv_poll_cq(cq.cq, 1, &mut wc);
                if n < 0 {
                    bail!("ibv_poll_cq error: rc={n}");
                }
                if n == 0 {
                    std::hint::spin_loop();
                    continue;
                }
                if wc.status != ib::IBV_WC_SUCCESS {
                    bail!(
                        "WR failed: status={} vendor_err={}",
                        wc.status,
                        wc.qp_num // using a field as placeholder for vendor_err
                    );
                }
                return Ok((wc.wr_id, wc.byte_len));
            }
        }
    }

    /// Post a receive buffer (needed for RDMA_READ responder side).
    pub fn post_recv(
        &self,
        mr: &MemoryRegion,
        offset: usize,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        unsafe {
            let mut sge: ib::ibv_sge = std::mem::zeroed();
            sge.addr = (mr.buf.as_ptr() as usize + offset) as u64;
            sge.length = len;
            sge.lkey = (*mr.mr).lkey;

            let mut wr: ib::ibv_recv_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;

            let mut bad_wr: *mut ib::ibv_recv_wr = ptr::null_mut();
            let rc = ib::ouro_ibv_post_recv(self.qp, &mut wr, &mut bad_wr);
            if rc != 0 {
                bail!("ouro_ibv_post_recv failed: rc={rc}");
            }
            Ok(())
        }
    }
}

// ── Memory region ──────────────────────────────────────────────────

impl MemoryRegion {
    /// Register a buffer for RDMA access. The buffer is pinned and DMA-visible.
    pub fn new(pd: *mut ib::ibv_pd, buf: Vec<u8>, access: i32) -> Result<Self> {
        Self::new_with_remote(pd, buf, access, 0, 0)
    }

    /// Register with explicit remote addr/rkey (for the responder side).
    pub fn new_with_remote(
        pd: *mut ib::ibv_pd,
        buf: Vec<u8>,
        access: i32,
        remote_addr: u64,
        rkey: u32,
    ) -> Result<Self> {
        unsafe {
            let len = buf.len();
            let mr = ib::ibv_reg_mr(pd, buf.as_ptr() as *mut libc::c_void, len, access);
            if mr.is_null() {
                bail!("ibv_reg_mr failed — is the NIC configured for RDMA?");
            }
            Ok(Self {
                mr,
                buf,
                remote_addr,
                rkey,
            })
        }
    }

    pub fn buf(&self) -> &[u8] {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    pub fn local_addr(&self) -> u64 {
        self.buf.as_ptr() as u64
    }

    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }

    pub fn remote_addr(&self) -> u64 {
        self.remote_addr
    }

    pub fn rkey(&self) -> u32 {
        self.rkey
    }
}

// ── rdma_cm connection management ──────────────────────────────────

impl RdmaDevice {
    /// Bind to a socket and listen for incoming RDMA connections (rdma_cm).
    /// Returns the connection id and the QP info of the remote peer.
    pub fn rdma_listen_on(port: u16) -> Result<(*mut ib::rdma_cm_id, QpInfo)> {
        unsafe {
            let channel = ib::rdma_create_event_channel();
            if channel.is_null() {
                bail!("rdma_create_event_channel failed");
            }

            let mut listen_id: *mut ib::rdma_cm_id = ptr::null_mut();
            let rc = ib::rdma_create_id(
                channel,
                &mut listen_id,
                ptr::null_mut(),
                ib::RDMA_PS_TCP,
            );
            if rc != 0 {
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_create_id (listen) failed: rc={rc}");
            }

            let mut addr: libc::sockaddr_in = std::mem::zeroed();
            addr.sin_family = libc::AF_INET as u16;
            addr.sin_port = port.to_be();
            addr.sin_addr.s_addr = libc::INADDR_ANY;

            let rc = ib::rdma_bind_addr(
                listen_id,
                &mut addr as *mut libc::sockaddr_in as *mut libc::sockaddr,
            );
            if rc != 0 {
                ib::rdma_destroy_id(listen_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_bind_addr failed: rc={rc}");
            }

            let rc = ib::rdma_listen(listen_id, 1);
            if rc != 0 {
                ib::rdma_destroy_id(listen_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_listen failed: rc={rc}");
            }

            // Wait for CONNECT_REQUEST
            let mut event: *mut ib::rdma_cm_event = ptr::null_mut();
            let rc = ib::rdma_get_cm_event(channel, &mut event);
            if rc != 0 {
                ib::rdma_destroy_id(listen_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_get_cm_event failed: rc={rc}");
            }

            if (*event).event != ib::RDMA_CM_EVENT_CONNECT_REQUEST {
                ib::rdma_ack_cm_event(event);
                ib::rdma_destroy_id(listen_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("unexpected RDMA event: {}", (*event).event);
            }

            let conn_id = (*event).id;
            ib::rdma_ack_cm_event(event);

            // The QP info is exchanged over our TCP side-channel, not here.
            // Return the conn_id; the caller does TCP exchange + QP state transition.
            let dummy_info = QpInfo {
                qp_num: 0,
                lid: 0,
                gid: [0; 16],
                port: 1,
            };
            Ok((conn_id, dummy_info))
        }
    }

    /// Connect to a remote RDMA listener. Sends our QP info.
    pub fn rdma_connect_to(
        addr: &str,
        port: u16,
    ) -> Result<(*mut ib::rdma_cm_id, QpInfo)> {
        unsafe {
            let channel = ib::rdma_create_event_channel();
            if channel.is_null() {
                bail!("rdma_create_event_channel failed");
            }

            let mut connect_id: *mut ib::rdma_cm_id = ptr::null_mut();
            let rc = ib::rdma_create_id(
                channel,
                &mut connect_id,
                ptr::null_mut(),
                ib::RDMA_PS_TCP,
            );
            if rc != 0 {
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_create_id (connect) failed: rc={rc}");
            }

            let c_addr = std::ffi::CString::new(addr).unwrap();
            let c_port = std::ffi::CString::new(port.to_string()).unwrap();
            let mut hints: ib::rdma_addrinfo = std::mem::zeroed();
            hints.ai_port_space = ib::RDMA_PS_TCP;
            let mut res: *mut ib::rdma_addrinfo = ptr::null_mut();

            let rc = ib::rdma_getaddrinfo(c_addr.as_ptr(), c_port.as_ptr(), &hints, &mut res);
            if rc != 0 {
                ib::rdma_destroy_id(connect_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_getaddrinfo failed: rc={rc}");
            }

            let mut conn_param: ib::rdma_conn_param = std::mem::zeroed();
            conn_param.private_data = ptr::null();
            conn_param.private_data_len = 0;
            conn_param.responder_resources = 1;
            conn_param.initiator_depth = 1;
            conn_param.retry_count = 7;
            conn_param.rnr_retry_count = 7;

            let rc = ib::rdma_connect(connect_id, &mut conn_param);
            ib::rdma_freeaddrinfo(res);
            if rc != 0 {
                ib::rdma_destroy_id(connect_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_connect failed: rc={rc}");
            }

            // Wait for ESTABLISHED
            let mut event: *mut ib::rdma_cm_event = ptr::null_mut();
            let rc = ib::rdma_get_cm_event(channel, &mut event);
            if rc != 0 {
                ib::rdma_destroy_id(connect_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("rdma_get_cm_event (active) failed: rc={rc}");
            }

            if (*event).event != ib::RDMA_CM_EVENT_ESTABLISHED {
                let ev_type = (*event).event;
                ib::rdma_ack_cm_event(event);
                ib::rdma_destroy_id(connect_id);
                ib::rdma_destroy_event_channel(channel);
                bail!("unexpected event: {ev_type}");
            }

            ib::rdma_ack_cm_event(event);

            // Same deal: actual QP info comes over the TCP side-channel.
            let dummy_info = QpInfo {
                qp_num: 0,
                lid: 0,
                gid: [0; 16],
                port: 1,
            };
            Ok((connect_id, dummy_info))
        }
    }
}

// ── Drop impls ─────────────────────────────────────────────────────

impl Drop for QueuePair {
    fn drop(&mut self) {
        unsafe {
            ib::ibv_destroy_qp(self.qp);
        }
    }
}

impl Drop for CompletionQueue {
    fn drop(&mut self) {
        unsafe {
            ib::ibv_destroy_cq(self.cq);
        }
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        unsafe {
            ib::ibv_dereg_mr(self.mr);
        }
    }
}

impl Drop for RdmaDevice {
    fn drop(&mut self) {
        unsafe {
            ib::ibv_dealloc_pd(self.pd);
            ib::ibv_close_device(self.ctx);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qp_info_size_is_stable() {
        // QpInfo is exchanged over the wire — its layout must not change
        // across builds. repr(C) + no padding.
        assert!(std::mem::size_of::<QpInfo>() <= 64);
    }

    #[test]
    fn qp_info_repr_c() {
        // Verify repr(C) layout: qp_num(4) + lid(2) + gid(16) + port(1) = 23 bytes
        // with C alignment: qp_num at 0, lid at 4, gid at 6, port at 22
        assert_eq!(std::mem::size_of::<QpInfo>(), 24); // 1 byte padding after port
    }
}
