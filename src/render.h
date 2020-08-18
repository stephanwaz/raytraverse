//
// Created by Stephen Wasilewski on 8/14/20.
//

#ifndef RAYTRAVERSE_RENDER_H
#define RAYTRAVERSE_RENDER_H
#include <pybind11/pybind11.h>

class Renderer {

protected:
    Renderer() = default;

public:
    ~Renderer() = default;
    virtual void initialize(PyObject* pyargv);
    virtual void initc(int argcount, char **argvector);
    virtual void call(char *fname);
    static Renderer& getInstance();
    static void resetInstance();
    static void resetRadiance();

protected:
    static Renderer* renderer;
    int nproc = 1;
    int argc = 0;
    char** argv = nullptr;
};


#endif //RAYTRAVERSE_RENDER_H
