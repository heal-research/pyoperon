{
  description = "pyoperon";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  inputs.operon.url = "github:heal-research/operon?rev=63c36289282067c898b8d1aa19f5d246656371da";
  inputs.pratt-parser.url = "github:foolnotion/pratt-parser-calculator?rev=a15528b1a9acfe6adefeb41334bce43bdb8d578c";
  inputs.vstat.url = "github:heal-research/vstat?rev=79b9ba2d69fe14e9e16a10f35d4335ffa984f02d";

  outputs = { self, flake-utils, nixpkgs, nur, operon, pratt-parser, vstat }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nur.overlay ];
          };
          repo = pkgs.nur.repos.foolnotion;
        in rec
        {
          defaultPackage = pkgs.gcc11Stdenv.mkDerivation {
            name = "pyoperon";
            src = self;

            cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];

            nativeBuildInputs = with pkgs; [ cmake ];

            buildInputs = with pkgs; [
                # python environment for bindings and scripting
                (python39.override { stdenv = gcc11Stdenv; })
                (python39.withPackages (ps: with ps; [ pybind11 numpy pandas scikit-learn requests ]))
                # Project dependencies and utils for profiling and debugging
                fmt
                ceres-solver
                operon.defaultPackage.${system}
                pratt-parser.defaultPackage.${system}
                vstat.defaultPackage.${system}

                # Some dependencies are provided by a NUR repo
                repo.aria-csv
                repo.fast_float
                repo.robin-hood-hashing
                repo.span-lite
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
              nativeBuildInputs = defaultPackage.nativeBuildInputs ++ (with pkgs; [ bear clang_13 clang-tools cppcheck ]);
              buildInputs = defaultPackage.buildInputs;

              shellHook = ''
                  PYTHONPATH=$PYTHONPATH:${defaultPackage.out}
                  LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
                  '';
          };
        }
      );
}
