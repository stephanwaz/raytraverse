//
// Created by Stephen Wasilewski on 8/14/20.
//

#ifndef RAYTRAVERSE_RENDER_H
#define RAYTRAVERSE_RENDER_H
#include <pybind11/pybind11.h>

class Renderer {

public:
    explicit Renderer(PyObject* pyargv);
    virtual void call(std::string fname);

protected:
    int nproc = 1;
    int argc = 0;
    char** argv = nullptr;
};



#endif //RAYTRAVERSE_RENDER_H
