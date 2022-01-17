{
  description = "pyoperon";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, nur }:
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

                # Some dependencies are provided by a NUR repo
                repo.aria-csv
                repo.fast_float
                repo.operon
                repo.pratt-parser
                repo.robin-hood-hashing
                repo.span-lite
                repo.taskflow
                repo.vectorclass
                repo.vstat
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
