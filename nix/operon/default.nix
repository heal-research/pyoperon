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
  fast_float,
  fmt_8,
  git,
  jemalloc,
  openlibm,
  pkg-config,
  pratt-parser,
  robin-hood-hashing,
  scnlib,
  span-lite,
  taskflow,
  vectorclass,
  vstat,
  xxhash,
  # build options
  useSinglePrecision ? true,
  buildCliPrograms ? false,
  enableShared ? !stdenv.hostPlatform.isStatic,
  useCeres ? false,
  useOpenLibm ? true,
  useJemalloc ? false
}:
stdenv.mkDerivation rec {
  pname = "operon";
  version = "0.3.1";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "operon";
    rev = "4777b3bc3ff99c42de0578b6f20fc0606ec9c084";
    sha256 = "sha256-ea9B+/Hg/MOaIgaX9AZG7+RlNJ7WQNqXlzKuGQX80b4=";
  };

  nativeBuildInputs = [ cmake git ];

  buildInputs = [
    aria-csv
    cpp-sort
    eigen
    vectorclass 
    fast_float
    git
    span-lite
    pkg-config
    pratt-parser
    robin-hood-hashing
    taskflow
    vectorclass
    vstat
    (scnlib.override { enableShared = enableShared; })
    (fmt_8.override { enableShared = enableShared; })
    (xxhash.override { enableShared = enableShared; })
  ] ++ lib.optionals buildCliPrograms [ cxxopts ]
    ++ lib.optionals useCeres [ ceres-solver ]
    ++ lib.optionals useOpenLibm [ openlibm ]
    ++ lib.optionals useJemalloc [ jemalloc ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DUSE_SINGLE_PRECISION=${if useSinglePrecision then "ON" else "OFF"}"
    "-DBUILD_CLI_PROGRAMS=${if buildCliPrograms then "ON" else "OFF"}"
    "-DBUILD_SHARED_LIBS=${if enableShared then "ON" else "OFF"}"
    "-DUSE_JEMALLOC=${if useJemalloc then "ON" else "OFF"}"
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
