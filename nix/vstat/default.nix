{ lib, stdenv, fetchFromGitHub, cmake, eve, pkg-config }:

stdenv.mkDerivation rec {
  pname = "vstat";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "vstat";
    rev = "1d9830770c423e9237f437ea4f0dc00f6630cb83";
    sha256 = "sha256-80O6oiNuyNKh34uRyulW12oGG35LnPcUwz3ttE/PG1w=";
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

