{
  description = "pyoperon";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/master";
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
        stdenv_ = pkgs.llvmPackages_17.stdenv;
        python_ = pkgs.python311;

        operon = pkgs.callPackage ./nix/operon {
          enableShared = enableShared;
          vstat = pkgs.callPackage ./nix/vstat { };
          lbfgs = lbfgs.packages.${system}.default;
          stdenv = stdenv_;
        };

        pyoperon = stdenv_.mkDerivation {
          name = "pyoperon";
          src = self;

          cmakeFlags = [
            "-DCMAKE_CXX_FLAGS=${if pkgs.hostPlatform.isx86_64 then "-g" else ""}"
          ];

          cmakeBuildType = "Debug";

          dontStrip = true;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            python_
            python_.pkgs.pybind11
          ];

          buildInputs = with pkgs; [
            python_.pkgs.setuptools
            python_.pkgs.wheel
            python_.pkgs.requests
            operon
          ] ++ operon.buildInputs;
        };
      in
      rec {
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
          buildInputs = pyoperon.buildInputs;
          shellHook = ''
            $SHELL
          '';
        };

        devShells.pyenv = stdenv_.mkDerivation {
          name = "pyoperon-dev";
          nativeBuildInputs = pyoperon.nativeBuildInputs;
          buildInputs = pyoperon.buildInputs ++ (with pkgs; [ gdb valgrind gcc13 ])
                          ++ (with python_.pkgs; [ scikit-build ] ) # cmake integration and release preparation
                          ++ (with python_.pkgs; [ numpy scikit-learn pandas ipdb sympy requests matplotlib optuna jax jaxlib-bin ])
                          ++ (with pkgs; [ (pmlb.override { pythonPackages = python_.pkgs; }) ]);
          shellHook = ''
            $SHELL
          '';
        };

        # backwards compatibility
        defaultPackage = packages.default;
        defaultShell = devShells.default;
      });
}
