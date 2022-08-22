{
  description = "pyoperon";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/master";

    operon.url = "github:heal-research/operon/cpp20";
    pratt-parser.url =
      "github:foolnotion/pratt-parser-calculator?rev=a15528b1a9acfe6adefeb41334bce43bdb8d578c";
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

  outputs = { self, flake-utils, mach-nix, nixpkgs, foolnotion, operon
    , pratt-parser, pypi-deps-db }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };
        python = "python39";
        mach = import mach-nix { inherit pkgs python; };

        pyEnv = mach.mkPython {

          requirements = ''
            scikit-learn
            sympy
            numpy
            pandas
            pmlb
            eli5
          '';

          ignoreDataOutdated = true;
        };

        pyoperon = mach.nixpkgs.stdenv.mkDerivation {
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
            pkg-config
            pyEnv.python
            pyEnv.python.pkgs.pybind11
          ];

          buildInputs = with pkgs; [
            eigen
            fmt
            openlibm
            # python environment for bindings and scripting
            pyEnv
            # flakes
            operon.packages.${system}.default
            pratt-parser.defaultPackage.${system}
            # foolnotion overlay
            fast_float
            robin-hood-hashing
          ];
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
          pyoperon-debug = pyoperon.overrideAttrs (old: {
            cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Debug" ]; });
        };

        devShells.default = mach.nixpkgs.mkShell {
          nativeBuildInputs = pyoperon.nativeBuildInputs;
          buildInputs = pyoperon.buildInputs ++ (with pkgs; [ gdb valgrind ]);
        };
      });
}
