//
// Created by Stephen Wasilewski on 8/14/20.
//

#ifndef RAYTRAVERSE_RTRACE_H
#define RAYTRAVERSE_RTRACE_H
#include "render.h"
#include <pybind11/pybind11.h>

class Rtrace : public Renderer {

private:
    Rtrace() = default;

public:
    static Renderer& getInstance();
    void initialize(pybind11::object pyargv11);
    void initc(int argc, char **argv) override;
    void call(char *fname) override;
    void updateOSpec(char *vs, char of);

};


#endif //RAYTRAVERSE_RTRACE_H
