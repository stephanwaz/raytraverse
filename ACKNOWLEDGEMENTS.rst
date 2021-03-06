Citation
--------
Either the latest or specific releases of this software are archived with a
DOI at zenodo. See: https://doi.org/10.5281/zenodo.4091318

Additionally, please cite this
`conference paper <http://simaud.org/2021/content.php?f=32.pdf>`_
for a description of the directional sampling and integration method:

    Stephen Wasilewski, Lars O. Grobe, Roland Schregle, Jan Wienold, and
    Marilyne Andersen. 2021. *Raytraverse: Navigating the Lightfield to
    Enhance Climate-Based Daylight Modeling*. In 2021 Proceedings of the
    Symposium on Simulation in Architecture and Urban Design.

Licence
-------

| Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
| This Source Code Form is subject to the terms of the Mozilla Public
| License, v. 2.0. If a copy of the MPL was not distributed with this
| file, You can obtain one at http://mozilla.org/MPL/2.0/.

Acknowledgements
----------------

Thanks to additional project collaborators and advisors Marilyne Andersen, Lars
Grobe, Roland Schregle, Jan Wienold, and Stephen Wittkopf

This software development was financially supported by the Swiss National
Science Foundation as part of the ongoing research project “Light fields in
climate-based daylight modeling for spatio-temporal glare assessment”
(SNSF_ #179067).

Software Credits
----------------

    - Raytraverse uses Radiance_
    - As well as all packages listed in the requirements.txt file,
      raytraverse relies heavily on the Python packages numpy_, scipy_, and
      for key parts of the implementation.
    - C++ bindings, including exposing core radiance functions as methods to
      the renderer classes are made with pybind11_
    - Installation and building from source uses cmake_ and scikit-build_
    - This package was created with Cookiecutter_ and the
      `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Radiance: https://www.radiance-online.org
.. _numpy: https://numpy.org/doc/stable/reference/
.. _scipy: https://docs.scipy.org/doc/scipy/reference/
.. _pybind11: https://pybind11.readthedocs.io/en/stable/index.html
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/
.. _SNSF: http://www.snf.ch/en/Pages/default.aspx
.. _cmake: https://cmake.org/cmake/help/latest/
