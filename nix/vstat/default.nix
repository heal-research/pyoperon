{ lib, stdenv, fetchFromGitHub, cmake, eve, pkg-config }:

stdenv.mkDerivation rec {
  pname = "vstat";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "vstat";
    rev = "4ed22ae344c6a2a6e4522ad8b2c40070dd760600";
    sha256 = "sha256-xcQlOU6YLxykNsWnfbobrV0YmT0I3e0itRNrwxkW3jw=";
  };

  nativeBuildInputs = [ cmake pkg-config ];

  buildInputs = [ eve ];

  meta = with lib; {
    description = "C++17 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation).";
    homepage = "https://github.com/heal-research/vstat";
    license = licenses.mit;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}

