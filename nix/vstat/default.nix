{ lib, stdenv, fetchFromGitHub, cmake, eve, pkg-config }:

stdenv.mkDerivation rec {
  pname = "vstat";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "heal-research";
    repo = "vstat";
    rev = "97af7dc11f6899472e49296a409467f4f163b589";
    sha256 = "sha256-DYIoh6R3K/Nwxeag6P/OOp701DmH0sXn9JPIbIHUFJA=";
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

