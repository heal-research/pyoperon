{ lib, stdenv, fetchFromGitHub, cmake, eve, pkg-config }:

stdenv.mkDerivation rec {
  pname = "vstat";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "vstat";
    rev = "06a8f15b22a0da523097f3fe500489c08a3ec086";
    hash = "sha256-zlKiMQBKLm66rj7xOWpfugVCbSEAPiHZinWRSeoV/w4=";
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
