{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.03";
  inputs.pyoperon.url = "github:heal-research/pyoperon";

  outputs = { self, nixpkgs, pyoperon }: {

    nixosConfigurations.container = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules =
        [ ({ pkgs, ... }: {
            boot.isContainer = true;

            # Let 'nixos-version --json' know about the Git revision
            # of this flake.
            system.configurationRevision = nixpkgs.lib.mkIf (self ? rev) self.rev;

            # Network configuration.
            networking.useDHCP = false;
            networking.firewall.allowedTCPPorts = [ 80 ];

            buildInputs = [ pyoperon.defaultPackage.${system} ];

          })
        ];
    };
  };
}
