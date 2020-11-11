Git Info
---------
this project is hosted in two places, a private repo (master branch) at:

    https://gitlab.enterpriselab.ch/lightfields/raytraverse

and a public repo (release branch) at:

    https://github.com/stephanwaz/raytraverse

the repo also depends on two submodules, to initialize run the following::

    git clone https://github.com/stephanwaz/raytraverse
    cd raytraverse
    git submodule init
    git submodule update --remote
    git -C src/Radiance config core.sparseCheckout true
    cp src/sparse-checkout .git/modules/src/Radiance/info/
    git submodule update --remote --force src/Radiance

after a "git pull" make sure you also run::

    git submodule update

to track with the latest commit used by raytraverse.
