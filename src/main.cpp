//
// Created by Stephen Wasilewski on 8/17/20.
//
#include <iostream>
#include "rtrace.hh"

int main(int argc, char** argv) {
  Renderer& rdr = Rtrace::getInstance();
  argc -= 1;
  char *inp = argv[argc];
  rdr.initc(argc, argv);
  rdr.call(inp);

}