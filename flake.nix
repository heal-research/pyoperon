{
  description = "pyoperon";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion, pratt-parser }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            foolnotion.overlay
            # next we need to override stdenv to lower the ABI requirements
            # (to be more compatible with older distros / python envs)
            #(final: prev: {
            #  glibc = prev.glibc.overrideAttrs (old: { version = "2.34"; });
            #})
          ];
        };
        enableShared = false;
        stdenv = pkgs.stdenv;
        python = pkgs.python39;

        operon = pkgs.callPackage ./nix/operon {
          enableShared = enableShared;
          useOpenLibm = false;
          vstat = pkgs.callPackage ./nix/vstat { };
        };

        pyoperon = stdenv.mkDerivation {
          name = "pyoperon";
          src = self;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_FLAGS=${
              if pkgs.hostPlatform.isx86_64 then "-march=x86-64-v3" else ""
            }"
          ];

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            python
            python.pkgs.pybind11
          ];

          buildInputs = with pkgs; [
            python.pkgs.poetry
            python.pkgs.setuptools
            python.pkgs.wheel
            operon
          ] ++ operon.buildInputs;
        };
      in rec {
        packages = {
          default = pyoperon;
          pyoperon-generic = pyoperon.overrideAttrs (old: {
            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              "-DCMAKE_CXX_FLAGS=${
                if pkgs.hostPlatform.isx86_64 then "-march=x86-64" else ""
              }"
            ];
          });
          pyoperon-debug = pyoperon.overrideAttrs
            (old: { cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Debug" ]; });
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = pyoperon.nativeBuildInputs;
          buildInputs = pyoperon.buildInputs ++ (with pkgs; [ gdb valgrind ]);

          shellHook = ''
          '';
        };
      });
}
