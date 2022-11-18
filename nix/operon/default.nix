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
  openlibm,
  pkg-config,
  pratt-parser,
  unordered_dense,
  scnlib,
  taskflow,
  vstat,
  xxhash_cpp,
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

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "operon";
    rev = "23e4a2c7771e7f4d49ea7b45facad4bdb7d6afc5";
    sha256 = "sha256-DWPt/+BKHdb3HhLM5stDAyuXjFZCoEACir8XosSnF0s=";
  };

  nativeBuildInputs = [ cmake git ];

  buildInputs = [
    aria-csv
    cpp-sort
    eigen
    eve
    fast_float
    git
    pkg-config
    pratt-parser
    unordered_dense
    taskflow
    vstat
    xxhash_cpp
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
    description = "Modern, fast, scalable C++ framework for symbolic regression.";
    homepage = "https://github.com/heal-research/operon";
    license = licenses.mit;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}
