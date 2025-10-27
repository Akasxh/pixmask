#include "pixmask/api.h"

#include <iostream>

int main() {
    pixmask::initialize();
    std::cout << "pixmask " << pixmask::version_string() << "\n";
    return 0;
}
