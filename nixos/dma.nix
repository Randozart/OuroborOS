# OuroborOS RDMA proof binary (ouro-dma) — Tier 4 PoC (DMA_ROADMAP.md).
#
# Builds the ouro-cluster package's binaries from the same workspace
# lockfile as the agent. The cluster crate links libibverbs + librdmacm
# (the DMA transport + the C shim for the static-inline verbs functions),
# so rdma-core must be in buildInputs: it supplies both the link libs and
# <infiniband/verbs.h> for the shim compile (cc crate).
{ lib, rustPlatform, rdma-core, src ? ../., cargoLockFile ? ../Cargo.lock, rev ? "unknown" }:

rustPlatform.buildRustPackage {
  pname = "ouro-dma";
  version = "0.1.0";

  inherit src;

  OURO_BUILD_REV = rev;

  cargoLock = {
    lockFile = cargoLockFile;
  };

  buildAndTestSubdir = "cluster";
  # verbs.h for the C shim (build.rs cc) + libibverbs/librdmacm for link
  buildInputs = [ rdma-core ];

  doCheck = false;

  meta = with lib; {
    description = "OuroborOS SoftRoCE proof-of-concept (RDMA server/bench)";
    license = with licenses; [ mit asl20 ];
    mainProgram = "ouro-dma";
  };
}
