#!/bin/bash

# Git Info
# ---------
#
# this repo depends on two submodules, to initialize run the following::
#
#     git clone https://github.com/stephanwaz/raytraverse
#     cd raytraverse
#     git submodule init
#     git submodule update --remote
#     git -C src/Radiance config core.sparseCheckout true
#     cp src/sparse-checkout .git/modules/src/Radiance/info/
#     git submodule update --remote --force src/Radiance
#
# after a "git pull" make sure you also run::
#
#     git submodule update
#
# to track with the latest commit used by raytraverse.

printf "\n#######################################################\n"
echo have you confirmed that directory is ready for release?
echo make clean
echo make docs
echo make coverage
echo update HISTORY.rst
echo git commit


printf "#######################################################\n\n"
echo if you are giving an explicit version have you run bumpversion?
printf "REMOVE 'dev' from version tag (find/replace)\n"
printf "#######################################################\n\n"

printf "\n#######################################################\n"
printf "if you want this version archived, make sure repository is\n"
printf "enabled: https://zenodo.org/account/settings/github/\n"
printf "then create a release on github\n"
printf "https://github.com/stephanwaz/raytraverse/releases\n"
printf "\n#######################################################\n"

echo -n "proceed to release (y/n)? "
read -r answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    clean=$(git status --porcelain --untracked-files=no | wc -l)
    if [ "$clean" -lt 1 ]; then
        if [[ $# == 1 && ($1 == "patch" || $1 == "minor" || $1 == "major" || $1 == v*.*.* || $1 == "continue") ]]; then
            git checkout release
            git merge master
            if [[ $1 == v*.*.* ]]; then
                git tag -a "$1" -m "tagged for release $1"
            elif [[ $1 == "continue" ]]; then
                echo "using current commit"
            else
                bumpversion --tag --commit "$1"
            fi
            make clean
            python -m build
            echo -n "ok to push (y/n)? "
            read -r answer
            if [ "$answer" != "${answer#[Yy]}" ] ;then
                twine upload dist/*.tar.gz dist/*.whl
                git push
                git checkout master
                git merge release
                git push
                tag="$(git tag | tail -1)"
                git push origin $tag
			else
				git status
			fi
        else
            echo usage: ./release.sh "[patch/minor/major]"
            echo usage: ./release.sh "vX.X.X (assumes bumpversion has already been run and vX.X.X matches)"
            echo usage: ./release.sh "continue (for picking up an aborted release, run after git commit --amend from master branch)"
        fi
    else
        echo working directory is not clean!
        git status --porcelain --untracked-files=no
    fi
else
    echo aborted
fi


