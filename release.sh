#!/bin/bash
clean=$(git status --porcelain --untracked-files=no | wc -l)
if [ "$clean" -lt 1 ]; then
    if [[ $# == 1 && ($1 == "patch" || $1 == "minor" || $1 == "major") ]]; then
        git checkout release
        git merge master
        bumpversion --tag --commit "$1"
        make dist
        make coverall
        twine upload dist/*.tar.gz
        git push --follow-tags
        git checkout master
        git merge release
        git push --follow-tags
    else
        echo usage: ./release.sh "[patch/minor/major]"
    fi
else
    echo working directory is not clean!
    git status --porcelain --untracked-files=no
fi
