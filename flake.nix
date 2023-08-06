{
  description = "pyoperon";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/staging-next";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
    lbfgs.url = "github:foolnotion/lbfgs";

    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    pratt-parser.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion, pratt-parser, lbfgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };
        enableShared = false;
        stdenv_ = pkgs.overrideCC pkgs.llvmPackages_16.stdenv (
          pkgs.clang_16.override { gccForLibs = pkgs.gcc13.cc; }
        );
        python = pkgs.python310;

        operon = pkgs.callPackage ./nix/operon {
          enableShared = enableShared;
          useOpenLibm = false;
          vstat = pkgs.callPackage ./nix/vstat { };
          lbfgs = lbfgs.packages.${system}.default;
          stdenv = stdenv_;
        };

        pyoperon = stdenv_.mkDerivation {
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
            python.pkgs.setuptools
            python.pkgs.wheel
            python.pkgs.requests
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

        devShells.default = stdenv_.mkDerivation {
          name = "pyoperon-dev";
          nativeBuildInputs = pyoperon.nativeBuildInputs;
          buildInputs = pyoperon.buildInputs ++ (with pkgs; [ gdb valgrind ])
                          ++ (with python.pkgs; [ scikit-build ] ) # cmake integration and release preparation
                          ++ (with python.pkgs; [ numpy scikit-learn pandas seaborn ipdb sympy requests ])
                          ++ (with pkgs; [ (pmlb.override { pythonPackages = python.pkgs; }) ]);
        };

        # backwards compatibility
        defaultPackage = packages.default;
        defaultShell = devShells.default;
      });
}
