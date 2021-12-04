#include <string>

#include "pyoperon/pyoperon.hpp"

auto main() -> int
{
  exported_class e;

  return std::string("pyoperon") == e.name() ? 0 : 1;
}
