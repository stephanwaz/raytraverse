/* Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
 * =======================================================================
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *=======================================================================
 *
 * The Following code is copied from, adapts, includes, and/or links parts of
 * the Radiance source code which is licensed by the following:
 *
 * The Radiance Software License, Version 1.0
 *
 * Copyright (c) 1990 - 2018 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory.   All rights reserved.
 *
 * If a copy of the full text of the License was not distributed with this file
 * (in ./ray/License.txt) the License is available at
 * https://www.radiance-online.org/download-install/license
 */
#include <iostream>
#include "rtrace.hh"

int main(int argc, char** argv) {
  Renderer& rdr = Rtrace::getInstance();
  argc -= 2;
  char *inp = argv[argc + 1];
  rdr.initc(argc, argv);
  rdr.loadscene(argv[argc]);
  rdr.loadsrc(NULL, 0);
//  std::cout << "call 1" << std::endl;
  rdr.call(inp);
  const char* srcname = "sun.rad";
  rdr.loadsrc(const_cast<char *>(srcname), -1);
  std::cout << "call 2" << std::endl;
  rdr.call(inp);
  const char* srcname2 = "sun2.rad";
  rdr.loadsrc(const_cast<char *>(srcname2), -1);
  std::cout << "call 3" << std::endl;
  rdr.call(inp);
  rdr.loadsrc(const_cast<char *>(srcname), 0);
  std::cout << "call 4" << std::endl;
  rdr.call(inp);
//  rdr.loadsrc(NULL, -1);
  const char* argv2[] = {"rtracemain", "-ab", "1", "-ad", "300", "-as", "150", "-n", "5", "-I"};
  rdr.initc(argc, const_cast<char **>(argv2));
  rdr.loadsrc(const_cast<char *>(srcname), -1);
  std::cout << "call 5" << std::endl;
  rdr.call(inp);
  rdr.initc(argc, argv);
  rdr.loadsrc(const_cast<char *>(srcname), -1);
  std::cout << "call 6" << std::endl;
  rdr.call(inp);

}
