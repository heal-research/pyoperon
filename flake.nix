{
  description = "pyoperon";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nur.url = "github:nix-community/NUR";
    nixpkgs.url = "github:nixos/nixpkgs/master";

    operon.url = "github:heal-research/operon/cpp20";
    pratt-parser.url = "github:foolnotion/pratt-parser-calculator?rev=a15528b1a9acfe6adefeb41334bce43bdb8d578c";
    pypi-deps-db.url = "github:DavHau/pypi-deps-db";

    mach-nix = {
      url = "github:DavHau/mach-nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
        pypi-deps-db.follows = "pypi-deps-db";
      };
    };
  };

  outputs =
    { self, flake-utils, mach-nix, nixpkgs, nur, operon, pratt-parser, pypi-deps-db }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
        repo = pkgs.nur.repos.foolnotion;
        mach = mach-nix.lib.${system};

        python-env = mach.mkPython {
          python = "python39";

          requirements = ''
            eli5
            joblib
            numpy
            optuna
            pandas
            pmlb
            pyyaml
            scikit-learn
            sympy
          '';

          ignoreDataOutdated = true;
        };
      in rec {
        defaultPackage = pkgs.gcc12Stdenv.mkDerivation {
          name = "pyoperon";
          src = self;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_FLAGS=${
              if pkgs.targetPlatform.isx86_64 then "-march=haswell" else ""
            }"
          ];

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            python-env.python
            python-env.python.pkgs.pybind11
          ];

          buildInputs = with pkgs; [
            eigen
            fmt
            openlibm
            # python environment for bindings and scripting
            python-env
            # flakes
            operon.defaultPackage.${system}
            pratt-parser.defaultPackage.${system}
            repo.fast_float
            repo.robin-hood-hashing
          ];
        };

        devShell = pkgs.gcc12Stdenv.mkDerivation {
          name = "pyoperon-dev";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
          nativeBuildInputs = defaultPackage.nativeBuildInputs;
          buildInputs = defaultPackage.buildInputs
            ++ (with pkgs; [ gdb valgrind ]);

          shellHook = ''
            PYTHONPATH=$PYTHONPATH:${defaultPackage.out}
          '';
        };
      });
}
