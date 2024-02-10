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
  fmt,
  git,
  lbfgs,
  mdspan,
  ned14-outcome,
  ned14-quickcpplib,
  ned14-status-code,
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
}:
stdenv.mkDerivation rec {
  pname = "operon";
  version = "0.3.1";

  #src = /home/bogdb/src/operon_tmp;

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "operon";
    rev = "b59cc849ff406b6157d45778c90ec4ca22e2944d";
    hash = "";
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
    mdspan
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
    (fmt.override { enableShared = enableShared; })
  ] ++ lib.optionals buildCliPrograms [ cxxopts ]
    ++ lib.optionals useCeres [ ceres-solver ];

  cmakeFlags = [
    "-DUSE_SINGLE_PRECISION=${if useSinglePrecision then "ON" else "OFF"}"
    "-DBUILD_CLI_PROGRAMS=${if buildCliPrograms then "ON" else "OFF"}"
    "-DBUILD_SHARED_LIBS=${if enableShared then "ON" else "OFF"}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=${if enableShared then "OFF" else "ON"}"
    "-DCMAKE_CXX_FLAGS=${if stdenv.targetPlatform.isx86_64 then "-g" else ""}"
  ];

  #cmakeBuildType = "Debug";
  #dontStrip = true;

  meta = with lib; {
    description = "Modern, fast, scalable C++ framework for symbolic regression";
    homepage = "https://github.com/heal-research/operon";
    license = licenses.mit;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}
