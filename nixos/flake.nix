{
  description = "OuroborOS node image — the OS is the agent (R2_BRINGUP.md §10)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # crates.io blocks curl's default User-Agent (403) under load;
      # their AUP requires an identifying UA. Inject one into every
      # fetchurl (the crate-tarball fetches go through bare fetchurl).
      # Transparent for functor/functional args (unstable's fetchurl
      # supports both call conventions).
      uaOverlay = final: prev: {
        fetchurl = args:
          if builtins.isFunction args then
            prev.fetchurl args
          else
            let opts = args.curlOptsList or [ ]; in
            prev.fetchurl (removeAttrs args [ "curlOptsList" ] // {
              curlOptsList = [
                "-A"
                "OuroborOS-node-image/1.0 (nix; contact: randozart@gmail.com)"
              ] ++ opts;
            });
      };

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ uaOverlay ];
      };

      ouro-agent = pkgs.callPackage ./agent.nix {
        src = ../.;
        cargoLockFile = ../Cargo.lock;
        # WP-U3: the build stamp travels with the binary (OURO_BUILD_REV)
        # and with the image (/etc/ouro/image-rev). Drift is visible.
        rev = self.shortRev or self.dirtyShortRev or "unknown";
      };

      # WP-DMA Tier 4: the SoftRoCE proof binary rides in the image —
      # the tail serves RDMA reads out of the box.
      ouro-dma = pkgs.callPackage ./dma.nix {
        src = ../.;
        cargoLockFile = ../Cargo.lock;
        rev = self.shortRev or self.dirtyShortRev or "unknown";
      };

      # WP-U4: the update trust anchor. The seed never leaves the head;
      # the public key is committed and baked into every image. A
      # missing or malformed key must fail the build, not ship silent.
      updatePubkey =
        let
          raw = builtins.readFile ../keys/update.signing.pub;
          stripped = builtins.replaceStrings [ "\n" "\r" " " ] [ "" "" "" ] raw;
        in
        assert builtins.stringLength stripped == 64;
        stripped;
    in
    {
      nixosConfigurations.ouro-node = nixpkgs.lib.nixosSystem {
        inherit system;
        specialArgs = {
          inherit ouro-agent ouro-dma;
          rev = self.shortRev or self.dirtyShortRev or "unknown";
          inherit updatePubkey;
        };
        modules = [
          (nixpkgs + "/nixos/modules/installer/cd-dvd/iso-image.nix")
          ./node-image.nix
        ];
      };

      packages.${system} = {
        ouro-agent = ouro-agent;
        ouro-dma = ouro-dma;
        node-image = self.nixosConfigurations.ouro-node.config.system.build.isoImage;
        default = self.packages.${system}.node-image;
      };
    };
}
