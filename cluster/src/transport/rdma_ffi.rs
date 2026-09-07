//! Minimal ibverbs + rdma_cm FFI bindings for the DMA proof-of-concept.
//!
//! `rdma-sys` 0.3.0 pins bindgen 0.59.2 which panics on anonymous unions
//! in `ib_user_ioctl_verbs.h` (kernel ≥6.10). We declare only the ~20
//! functions we actually use — avoids the entire bindgen pipeline.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type c_int = libc::c_int;
pub type c_uint = libc::c_uint;

// ── opaque handles (with minimal fields we access) ────────────────
pub enum ibv_context_opaque {}
pub enum ibv_pd_opaque {}
pub enum ibv_cq_opaque {}
pub enum ibv_qp_opaque {}
pub enum ibv_device_opaque {}
pub type ibv_device = ibv_device_opaque;

pub type ibv_cq = ibv_cq_opaque;
pub type ibv_qp = ibv_qp_opaque;

/// ibv_context — we access `.device` (pointer to ibv_device).
#[repr(C)]
pub struct ibv_context {
    pub device: *mut ibv_device,
}

pub type ibv_pd = ibv_pd_opaque;

/// ibv_mr — we access `.lkey`.
#[repr(C)]
pub struct ibv_mr {
    pub addr: *mut c_void,
    pub length: usize,
    pub lkey: u32,
    pub rkey: u32,
}

// ── ibv_gid ───────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_gid {
    pub raw: [u8; 16],
}

// ── ibv_device_attr ───────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_device_attr {
    pub fw_ver: [u8; 64],
    pub node_guid: u64,
    pub sys_image_guid: u64,
    pub max_mr_size: u64,
    pub page_size_cap: u64,
    pub vendor_id: u32,
    pub vendor_part_id: u32,
    pub hw_ver: u32,
    pub max_qp: c_int,
    pub max_qp_wr: c_int,
    pub max_sge: c_int,
    pub max_sge_rd: c_int,
    pub max_cq: c_int,
    pub max_cqe: c_int,
    pub max_mr: c_int,
    pub max_pd: c_int,
    pub max_qp_rd_atom: c_int,
    pub max_ee_rd_atom: c_int,
    pub max_res_rd_atom: c_int,
    pub max_qp_init_rd_atom: c_int,
    pub max_ee_init_rd_atom: c_int,
    pub atomic_cap: c_int,
    pub max_ee: c_int,
    pub max_rdd: c_int,
    pub max_mw: c_int,
    pub max_raw_ipv6_qp: c_int,
    pub max_raw_ethy_qp: c_int,
    pub max_mcast_grp: c_int,
    pub max_mcast_qp_attach: c_int,
    pub max_mcast_total: c_int,
    pub max_ah: c_int,
    pub max_fmr: c_int,
    pub max_map_per_fmr: c_int,
    pub max_srq: c_int,
    pub max_srq_wr: c_int,
    pub max_srq_sge: c_int,
    pub max_pkeys: u16,
    pub local_ca_ack_delay: u8,
    pub phys_port_cnt: u8,
}

// ── ibv_port_attr ─────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_port_attr {
    pub state: c_int,       // ibv_port_state
    pub max_mtu: c_int,     // ibv_mtu
    pub active_mtu: c_int,
    pub gid_tbl_len: c_int,
    pub port_cap_flags: u32,
    pub max_msg_sz: u32,
    pub bad_pkey_cntr: u32,
    pub qkey_viol_cntr: u32,
    pub pkey_table_len: u16,
    pub lid: u16,
    pub sm_lid: u16,
    pub lmc: u8,
    pub max_vl_num: u8,
    pub sm_sl: u8,
    pub subnet_timeout: u8,
    pub init_type_reply: u8,
    pub active_width: u8,
    pub active_speed: u8,
    pub phys_state: u8,
    pub link_layer: u8,
    pub flags: u8,
}

// ── ibv_pd (with fields for query) ───────────────────────────────
#[repr(C)]
pub struct ibv_pd_attr {
    pub pd_context: u64,
    pub pd_handle: u32,
    pub opaque_handle: u32,
    pub max_mw: u32,
    pub maxrän: u32,
    pub max_sge: u32,
    pub max_recv_wr: u32,
    pub max_send_wr: u32,
    pub max_inline_data: u32,
    pub num_comp_vectors: u32,
}

// ── ibv_qp_init_attr ──────────────────────────────────────────────
#[repr(C)]
pub struct ibv_qp_init_attr {
    pub send_cq: *mut ibv_cq,
    pub recv_cq: *mut ibv_cq,
    pub srq: *mut c_void, // *mut ibv_srq (unused)
    pub qp_context: *mut c_void,
    pub cap: ibv_qp_cap,
    pub qp_type: c_int,  // ibv_qp_type
    pub sq_sig_all: c_int,
}

// ── ibv_qp_cap ────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_qp_cap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

// ── ibv_qp_attr ───────────────────────────────────────────────────
#[repr(C)]
pub struct ibv_qp_attr {
    pub qp_state: c_int,
    pub cur_qp_state: c_int,
    pub path_mtu: c_int,
    pub path_mig_state: c_int,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: c_int,
    pub ah_attr: ibv_ah_attr,
    pub alt_ah_attr: ibv_ah_attr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    pub rate_limit: u32,
    pub access_flags: c_int,
}

// ── ibv_ah_attr ───────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_ah_attr {
    pub grh: ibv_global_route,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub is_global: u8,
    pub rate_limit: u8,
    pub port_num: u8,
    pub static_rate: u8,
    pub traffic_class: u8,
}

// ── ibv_global_route ──────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_global_route {
    pub dgid: ibv_gid,
    pub flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
}

// ── ibv_sge ───────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_sge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

// ── ibv_send_wr ───────────────────────────────────────────────────
#[repr(C)]
pub struct ibv_send_wr {
    pub wr_id: u64,
    pub next: *mut ibv_send_wr,
    pub sg_list: *mut ibv_sge,
    pub num_sge: c_int,
    pub opcode: c_int,
    pub send_flags: c_int,
    pub imm_data_invalidated_rkey: imm_data_invalidated_rkey_union,
    pub wr: wr_t_union,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union imm_data_invalidated_rkey_union {
    pub imm_data: u32,
    pub invalidated_rkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union wr_t_union {
    pub rdma: ibv_rdma_wr,
    pub atomic: ibv_atomic_wr,
    pub ud: ibv_ud_wr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_rdma_wr {
    pub remote_addr: u64,
    pub rkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_atomic_wr {
    pub remote_addr: u64,
    pub compare_add: u64,
    pub swap: u64,
    pub rkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_ud_wr {
    pub ah: *mut c_void, // *mut ibv_ah
    pub remote_qpn: u32,
    pub remote_qkey: u32,
}

// ── ibv_recv_wr ───────────────────────────────────────────────────
#[repr(C)]
pub struct ibv_recv_wr {
    pub wr_id: u64,
    pub next: *mut ibv_recv_wr,
    pub sg_list: *mut ibv_sge,
    pub num_sge: c_int,
}

// ── ibv_wc (completion queue entry) ───────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_wc {
    pub wr_id: u64,
    pub status: c_int,
    pub opcode: c_int,
    pub byte_len: u32,
    pub imm_data_invalidated_rkey: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub dlid_path_bits: u8,
    pub port_num: u8,
}

// ── ibv_mr (memory region) ────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_mr_info {
    pub addr: *mut c_void,
    pub length: usize,
    pub lkey: u32,
    pub rkey: u32,
}

// ── rdma_cm types ─────────────────────────────────────────────────
pub enum rdma_cm_id_opaque {}
pub type rdma_cm_id = rdma_cm_id_opaque;

#[repr(C)]
pub struct rdma_cm_event {
    pub id: *mut rdma_cm_id,
    pub status: c_int,
    pub event: c_int,
    pub param: rdma_cm_event_param,
}

#[repr(C)]
pub struct rdma_cm_event_param {
    pub conn: rdma_cm_conn_param_out,
}

#[repr(C)]
pub struct rdma_cm_conn_param_out {
    pub private_data: *const c_void,
    pub private_data_len: u32,
    pub responder_resources: u8,
    pub initiator_depth: u8,
    pub flow_control: u8,
    pub retry_count: u8,
    pub rnr_retry_count: u8,
    pub srq: u8,
    pub qp_num: u32,
    pub qkey: u32,
}

pub enum rdma_event_channel_opaque {}
pub type rdma_event_channel = rdma_event_channel_opaque;

#[repr(C)]
pub struct rdma_conn_param {
    pub private_data: *const c_void,
    pub private_data_len: u32,
    pub responder_resources: u8,
    pub initiator_depth: u8,
    pub flow_control: u8,
    pub retry_count: u8,
    pub rnr_retry_count: u8,
    pub srq: u8,
    pub qp_num: u32,
    pub qkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rdma_addrinfo {
    pub ai_flags: c_int,
    pub ai_family: c_int,
    pub ai_socktype: c_int,
    pub ai_protocol: c_int,
    pub ai_src_len: usize,
    pub ai_dst_len: usize,
    pub ai_src_addr: *mut sockaddr,
    pub ai_dst_addr: *mut sockaddr,
    pub ai_next: *mut rdma_addrinfo,
    pub ai_port_space: c_int,
}

pub type sockaddr = libc::sockaddr;

// ── enum constants ────────────────────────────────────────────────

// ibv_qp_type
pub const IBV_QPT_RC: c_int = 2;

// ibv_qp_state
pub const IBV_QPS_RESET: c_int = 0;
pub const IBV_QPS_RTR: c_int = 3;
pub const IBV_QPS_RTS: c_int = 4;

// ibv_mtu
pub const IBV_MTU_1024: c_int = 2;

// ibv_wr_opcode
pub const IBV_WR_RDMA_READ: c_int = 0;
pub const IBV_WR_RDMA_WRITE: c_int = 1;
pub const IBV_WR_SEND: c_int = 2;

// ibv_send_flags
pub const IBV_SEND_SIGNALED: c_int = 1;

// ibv_access_flags
pub const IBV_ACCESS_LOCAL_WRITE: c_int = 1;
pub const IBV_ACCESS_REMOTE_WRITE: c_int = 8;
pub const IBV_ACCESS_REMOTE_READ: c_int = 4;

// ibv_wc_status
pub const IBV_WC_SUCCESS: c_int = 0;

// ibv_qp_attr_mask
pub const IBV_QP_STATE: c_int = 1;
pub const IBV_QP_PKEY_INDEX: c_int = 2;
pub const IBV_QP_PORT: c_int = 4;
pub const IBV_QP_ACCESS_FLAGS: c_int = 8;
pub const IBV_QP_AV: c_int = 64;
pub const IBV_QP_PATH_MTU: c_int = 128;
pub const IBV_QP_DEST_QPN: c_int = 256;
pub const IBV_QP_RQ_PSN: c_int = 512;
pub const IBV_QP_MAX_DEST_RD_ATOMIC: c_int = 1024;
pub const IBV_QP_MIN_RNR_TIMER: c_int = 2048;
pub const IBV_QP_TIMEOUT: c_int = 8192;
pub const IBV_QP_RETRY_CNT: c_int = 16384;
pub const IBV_QP_RNR_RETRY: c_int = 32768;
pub const IBV_QP_SQ_PSN: c_int = 65536;
pub const IBV_QP_MAX_QP_RD_ATOMIC: c_int = 262144;

// rdma_port_space
pub const RDMA_PS_TCP: c_int = 0x0100;

// rdma_cm_event_type
pub const RDMA_CM_EVENT_CONNECT_REQUEST: c_int = 1;
pub const RDMA_CM_EVENT_ESTABLISHED: c_int = 4;

// ── extern functions (linked at compile time) ─────────────────────
extern "C" {
    // Device management
    pub fn ibv_get_device_list(num_devices: *mut c_int) -> *mut *mut ibv_device;
    pub fn ibv_get_device_name(device: *const ibv_device) -> *const libc::c_char;
    pub fn ibv_get_device_guid(device: *const ibv_device) -> ibv_gid;
    pub fn ibv_open_device(device: *mut ibv_device) -> *mut ibv_context;
    pub fn ibv_close_device(context: *mut ibv_context) -> c_int;
    pub fn ibv_free_device_list(list: *mut *mut ibv_device);

    // Protection domain
    pub fn ibv_alloc_pd(context: *mut ibv_context) -> *mut ibv_pd;
    pub fn ibv_dealloc_pd(pd: *mut ibv_pd) -> c_int;

    // Query
    pub fn ibv_query_device(context: *mut ibv_context, device_attr: *mut ibv_device_attr) -> c_int;
    pub fn ibv_query_port(
        context: *mut ibv_context,
        port_num: u8,
        port_attr: *mut ibv_port_attr,
    ) -> c_int;
    pub fn ibv_query_gid(
        context: *mut ibv_context,
        port_num: u8,
        index: c_uint,
        gid: *mut ibv_gid,
    ) -> c_int;

    // CQ
    pub fn ibv_create_cq(
        context: *mut ibv_context,
        cqe: c_int,
        cq_context: *mut c_void,
        channel: *mut c_void,
        comp_vector: c_int,
    ) -> *mut ibv_cq;
    pub fn ibv_destroy_cq(cq: *mut ibv_cq) -> c_int;

    // QP
    pub fn ibv_create_qp(pd: *mut ibv_pd, qp_init_attr: *mut ibv_qp_init_attr) -> *mut ibv_qp;
    pub fn ibv_destroy_qp(qp: *mut ibv_qp) -> c_int;
    pub fn ibv_modify_qp(
        qp: *mut ibv_qp,
        attr: *mut ibv_qp_attr,
        attr_mask: c_int,
    ) -> c_int;
    pub fn ibv_query_qp(
        qp: *mut ibv_qp,
        attr: *mut ibv_qp_attr,
        attr_mask: c_int,
        qp_init_attr: *mut ibv_qp_init_attr,
    ) -> c_int;

    // Memory region
    pub fn ibv_reg_mr(
        pd: *mut ibv_pd,
        addr: *mut c_void,
        length: usize,
        access: c_int,
    ) -> *mut ibv_mr;
    pub fn ibv_dereg_mr(mr: *mut ibv_mr) -> c_int;

    // Send / Receive / Poll — these are static inlines in the header that
    // dispatch through qp->context->ops / cq->ops.  Our C shim (ibv_poll_cq_shim.c)
    // compiles against the real header and exports linkable symbols.
    pub fn ouro_ibv_poll_cq(cq: *mut ibv_cq, num_entries: c_int, wc: *mut ibv_wc) -> c_int;
    pub fn ouro_ibv_post_send(
        qp: *mut ibv_qp,
        wr: *mut ibv_send_wr,
        bad_wr: *mut *mut ibv_send_wr,
    ) -> c_int;
    pub fn ouro_ibv_post_recv(
        qp: *mut ibv_qp,
        wr: *mut ibv_recv_wr,
        bad_wr: *mut *mut ibv_recv_wr,
    ) -> c_int;

    // rdma_cm
    pub fn rdma_create_event_channel() -> *mut rdma_event_channel;
    pub fn rdma_destroy_event_channel(channel: *mut rdma_event_channel);
    pub fn rdma_create_id(
        channel: *mut rdma_event_channel,
        id: *mut *mut rdma_cm_id,
        context: *mut c_void,
        ps: c_int,
    ) -> c_int;
    pub fn rdma_destroy_id(id: *mut rdma_cm_id) -> c_int;
    pub fn rdma_bind_addr(id: *mut rdma_cm_id, addr: *mut sockaddr) -> c_int;
    pub fn rdma_listen(id: *mut rdma_cm_id, backlog: c_int) -> c_int;
    pub fn rdma_get_cm_event(
        channel: *mut rdma_event_channel,
        event: *mut *mut rdma_cm_event,
    ) -> c_int;
    pub fn rdma_ack_cm_event(event: *mut rdma_cm_event) -> c_int;
    pub fn rdma_getaddrinfo(
        node: *const libc::c_char,
        service: *const libc::c_char,
        hints: *const rdma_addrinfo,
        res: *mut *mut rdma_addrinfo,
    ) -> c_int;
    pub fn rdma_freeaddrinfo(res: *mut rdma_addrinfo);
    pub fn rdma_connect(id: *mut rdma_cm_id, conn_param: *mut rdma_conn_param) -> c_int;
}

// Helper to extract QP number from ibv_mr (just use the lkey offset).
// Actually we need to query QP to get its number — see ibv_query_qp.
