//
// Created by Stephen Wasilewski on 8/14/20.
//

#ifndef RAYTRAVERSE_RTRACE_H
#define RAYTRAVERSE_RTRACE_H
#include "render.h"
#include <pybind11/pybind11.h>

class Rtrace : public Renderer {

public:
    explicit Rtrace(pybind11::object pyargv11);
    void call(std::string fname) override;

};


#endif //RAYTRAVERSE_RTRACE_H
