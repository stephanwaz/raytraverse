//
// Created by Stephen Wasilewski on 8/14/20.
//

#ifndef RAYTRAVERSE_RCONTRIB_HH
#define RAYTRAVERSE_RCONTRIB_HH
#include "render.h"
#include <pybind11/pybind11.h>

class Rcontrib : public Renderer {

private:
    Rcontrib() = default;
    static Rcontrib* renderer;

public:
    static Rcontrib& getInstance();
    void initialize(pybind11::object pyargv11);
    void initc(int argc, char **argv) override;
    void call(char *fname) override;
    static void resetRadiance();
    static void resetInstance();

};


#endif //RAYTRAVERSE_RCONTRIB_HH
