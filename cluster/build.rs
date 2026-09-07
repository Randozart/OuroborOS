fn main() {
    // Link ibverbs and rdmacm for the DMA proof-of-concept.
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=rdmacm");

    // Compile the ibv shim (ibv_poll_cq, ibv_post_send, ibv_post_recv
    // are static inlines in the header — they dispatch through ops vtables).
    let out_dir = std::env::var("OUT_DIR").unwrap();
    cc::Build::new()
        .file("src/transport/ibv_poll_cq_shim.c")
        .compile("ibv_poll_cq_shim");
    println!("cargo:rustc-link-search=native={out_dir}");
}
