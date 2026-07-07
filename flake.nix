{
  description = "PyOperon development environment";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    foolnotion.url = "github:foolnotion/nur-pkg";
    infix-parser.url = "github:foolnotion/infix-parser";
    lbfgs.url = "github:foolnotion/lbfgs";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    operon.url = "github:heal-research/operon";
    vstat.url = "github:heal-research/vstat";

    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
    infix-parser.inputs.nixpkgs.follows = "nixpkgs";
    lbfgs.inputs.nixpkgs.follows = "nixpkgs";
    operon.inputs.nixpkgs.follows = "nixpkgs";
    vstat.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      foolnotion,
      infix-parser,
      lbfgs,
      vstat,
      operon,
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              foolnotion.overlay
              (final: prev: {
                vstat = vstat.packages.${system}.default;
              })
            ];
          };
          enableShared = false;
          stdenv_ = pkgs.llvmPackages_latest.stdenv;
          python_ = pkgs.python3;
          operon_ =
            if enableShared then
              operon.packages.${system}.library
            else
              operon.packages.${system}.library-static;

          pyoperon = stdenv_.mkDerivation {
            name = "pyoperon";
            src = self;

            cmakeFlags = [
              "--preset ${if pkgs.stdenv.hostPlatform.isx86_64 then "build-linux" else "build-osx"}"
            ];

            nativeBuildInputs = with pkgs; [
              act
              cmake
              ninja
              pkg-config
              python_
              python_.pkgs.pybind11
            ];

            # operon's public/header-visible deps (eve, vstat, fmt, etc.) are
            # propagatedBuildInputs on the operon derivation, so they come
            # along automatically via operon_ below. infix-parser's FastFloat
            # dependency is a private (non-propagated) buildInput on its own
            # derivation, but its installed CMake config still `find_dependency`s
            # FastFloat, so downstream consumers need it in scope explicitly.
            buildInputs =
              with pkgs;
              [
                fast-float
                foonathan-lexy
                (python_.withPackages (
                  ps: with ps; [
                    setuptools
                    wheel
                    requests
                    nanobind
                  ]
                ))
                (operon_.overrideAttrs (old: {
                  cmakeFlags = old.cmakeFlags ++ [ "-DUSE_SINGLE_PRECISION=1" ];
                }))
              ];
          };
        in
        rec {
          packages = {
            default = pyoperon;
            pyoperon-generic = pyoperon.overrideAttrs (old: {
              cmakeFlags = [
                "-DCMAKE_BUILD_TYPE=Release"
                "-DCMAKE_CXX_FLAGS=${if pkgs.hostPlatform.isx86_64 then "-march=x86-64" else ""}"
              ];
            });
            pyoperon-debug = pyoperon.overrideAttrs (old: {
              cmakeBuildType = "Debug";
            });
          };

          devShells.default = stdenv_.mkDerivation {
            name = "pyoperon-dev";
            nativeBuildInputs = pyoperon.nativeBuildInputs;
            buildInputs = pyoperon.buildInputs;
          };

          devShells.pyenv = stdenv_.mkDerivation {
            name = "pyoperon-dev";
            nativeBuildInputs = pyoperon.nativeBuildInputs;
            buildInputs =
              pyoperon.buildInputs
              ++ (with pkgs; [
                pdm
                virtualenv
                zlib
              ])
              ++ (with python_.pkgs; [
                numpy
                pandas
                scikit-learn
                pytest
              ]);
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/";
          };
        };
    };
}
