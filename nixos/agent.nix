# OuroborOS node agent — pure-Rust build, no bitnet-cpp.
#
# The R2 role (stage_setup/step/token/sample) runs on ouro_cluster::infer,
# which is pure Rust: the `bitnet` feature (C++ llama.cpp via bindgen,
# musl-hostile) is not needed on image nodes. Single static binary.
#
# NOTE (WP7 gate): cargoLock uses the workspace lockfile; the optional
# bitnet-rs/ouro-wgpu members still appear in it and must resolve in the
# vendored deps. If vendoring fights the workspace layout, pin a minimal
# lockfile for the image build instead of the workspace one.
{ lib, rustPlatform, src ? ../., cargoLockFile ? ../Cargo.lock }:

rustPlatform.buildRustPackage {
  pname = "ouro-agent";
  version = "0.1.0";

  inherit src;

  cargoLock = {
    lockFile = cargoLockFile;
  };

  buildAndTestSubdir = "agent";
  buildNoDefaultFeatures = true;

  # tests run in the sandbox without model files; the bitnet-gated ones
  # are compiled out with no-default-features
  doCheck = false;

  meta = with lib; {
    description = "OuroborOS node agent (getty-shim stdio mode)";
    license = with licenses; [ mit asl20 ];
    mainProgram = "ouro-agent";
  };
}
