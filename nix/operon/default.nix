{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  cpp-sort,
  # build inputs
  aria-csv,
  ceres-solver,
  cxxopts,
  eigen,
  eve,
  fast_float,
  fmt_9,
  git,
  lbfgs,
  ned14-outcome,
  ned14-quickcpplib,
  ned14-status-code,
  openlibm,
  pkg-config,
  pratt-parser,
  unordered_dense,
  scnlib,
  taskflow,
  vstat,
  xxHash,
  # build options
  useSinglePrecision ? true,
  buildCliPrograms ? false,
  enableShared ? !stdenv.hostPlatform.isStatic,
  useCeres ? false,
  useOpenLibm ? true,
}:
stdenv.mkDerivation rec {
  pname = "operon";
  version = "0.3.1";

  #src = /home/bogdb/src/operon-mdl-fix;

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "operon";
    rev = "e9f236749fbaa3480bfe27d064f42b229681a88d";
    hash = "sha256-9PGu3Bdh/KUG2LD3wXYmI8I+E5s5I1YEpJEOVM1wbQw=";
  };

  nativeBuildInputs = [ cmake git ];

  buildInputs = [
    aria-csv
    cpp-sort
    eigen
    eve
    fast_float
    git
    lbfgs
    ned14-outcome
    ned14-quickcpplib
    ned14-status-code
    pkg-config
    pratt-parser
    taskflow
    unordered_dense
    vstat
    xxHash
    (scnlib.override { enableShared = enableShared; })
    (fmt_9.override { enableShared = enableShared; })
  ] ++ lib.optionals buildCliPrograms [ cxxopts ]
    ++ lib.optionals useCeres [ ceres-solver ]
    ++ lib.optionals useOpenLibm [ openlibm ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DUSE_SINGLE_PRECISION=${if useSinglePrecision then "ON" else "OFF"}"
    "-DBUILD_CLI_PROGRAMS=${if buildCliPrograms then "ON" else "OFF"}"
    "-DBUILD_SHARED_LIBS=${if enableShared then "ON" else "OFF"}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=${if enableShared then "OFF" else "ON"}"
    "-DUSE_OPENLIBM=${if useOpenLibm then "ON" else "OFF"}"
    "-DCMAKE_CXX_FLAGS=${if stdenv.targetPlatform.isx86_64 then "-march=x86-64-v3" else ""}"
  ];

  meta = with lib; {
    description = "Modern, fast, scalable C++ framework for symbolic regression";
    homepage = "https://github.com/heal-research/operon";
    license = licenses.mit;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}
