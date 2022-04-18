{
  description = "pyoperon";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  inputs.operon.url = "github:heal-research/operon/cpp20";
  inputs.pratt-parser.url = "github:foolnotion/pratt-parser-calculator?rev=a15528b1a9acfe6adefeb41334bce43bdb8d578c";
  inputs.vstat.url = "github:heal-research/vstat/cpp20-eve";

  outputs = { self, flake-utils, nixpkgs, nur, operon, pratt-parser, vstat }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
        repo = pkgs.nur.repos.foolnotion;

        python = pkgs.python39.override { stdenv = pkgs.gcc11Stdenv; };
      in rec {
        defaultPackage = pkgs.gcc11Stdenv.mkDerivation {
          name = "pyoperon";
          src = self;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_FLAGS=${if pkgs.targetPlatform.isx86_64 then "-march=haswell" else ""}"
          ];

          nativeBuildInputs = with pkgs; [
            cmake
            python
          ];

          buildInputs = with pkgs; [
            # python environment for bindings and scripting
            (python.withPackages (ps:
              with ps; [
#                jupyterlab
                numpy
                pandas
                pybind11
                requests
                scikit-learn
                seaborn
              ]))
            # Project dependencies and utils for profiling and debugging
            eigen
            fmt
            openlibm
            pkg-config
            # flakes
            operon.defaultPackage.${system}
            pratt-parser.defaultPackage.${system}
            vstat.defaultPackage.${system}
            # Some dependencies are provided by a NUR repo
            repo.aria-csv
            repo.eve
            repo.fast_float
            repo.robin-hood-hashing
            repo.taskflow
            repo.vectorclass
            repo.xxhash
            # Needed for the example
            repo.eli5
            repo.pmlb
          ];
        };

        devShell = pkgs.gcc11Stdenv.mkDerivation {
          name = "pyoperon-dev";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
          nativeBuildInputs = defaultPackage.nativeBuildInputs;
          buildInputs = defaultPackage.buildInputs ++ (with pkgs; [ gdb valgrind ]);

          shellHook = ''
            PYTHONPATH=$PYTHONPATH:${defaultPackage.out}
            LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]}:$CMAKE_LIBRARY_PATH;
          '';
        };
      });
}
