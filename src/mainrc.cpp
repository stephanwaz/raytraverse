//
// Created by Stephen Wasilewski on 8/17/20.
//
#include <iostream>
#include "rcontrib.hh"

void test_run(int argc, char** argv, char* inp){
  Renderer& rdr = Rcontrib::getInstance();
  rdr.call(inp);
}

int main(int argc, char** argv) {
  Rcontrib& rdr = Rcontrib::getInstance();
  argc -= 1;
  char *inp = argv[argc];
  rdr.initc(argc, argv);
  rdr.call(inp);
  rdr.call(inp);
  test_run(argc, argv, inp);
  rdr.resetInstance();

}