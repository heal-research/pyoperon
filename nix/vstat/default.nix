{ lib, stdenv, fetchFromGitHub, cmake, vectorclass, pkg-config }:

stdenv.mkDerivation rec {
  pname = "vstat";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "vstat";
    rev = "c1c15c0be911d0fc7a4f9a4cd645977202356812";
    sha256 = "sha256-7ThxmL1+sYrBixFenlZd7wT7if9ziXEdN4eRQrNe994=";
  };

  nativeBuildInputs = [ cmake pkg-config ];

  buildInputs = [ vectorclass ];

  meta = with lib; {
    description = "C++17 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation).";
    homepage = "https://github.com/heal-research/vstat";
    license = licenses.mit;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}

