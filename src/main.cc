#include <iostream>
#include <fenv.h>

#include "poincare.h"
#include "args.h"


using namespace poincare;

int main(int argc, char** argv) {
    feenableexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_DIVBYZERO | FE_INVALID);
    std::vector<std::string> args(argv, argv + argc);
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->parse_args(args);
    Poincare poincare(a);
    poincare.train();
    poincare.save_vectors(a->output_vectors);
    return 0;
}
