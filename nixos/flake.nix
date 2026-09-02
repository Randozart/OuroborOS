{
  description = "OurobourOS node image — the OS is the agent (R2_BRINGUP.md §10)";

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
                "OurobourOS-node-image/1.0 (nix; contact: randozart@gmail.com)"
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
      };
    in
    {
      nixosConfigurations.ouro-node = nixpkgs.lib.nixosSystem {
        inherit system;
        specialArgs = { inherit ouro-agent; };
        modules = [
          (nixpkgs + "/nixos/modules/installer/cd-dvd/iso-image.nix")
          ./node-image.nix
        ];
      };

      packages.${system} = {
        ouro-agent = ouro-agent;
        node-image = self.nixosConfigurations.ouro-node.config.system.build.isoImage;
        default = self.packages.${system}.node-image;
      };
    };
}
