#!/bin/bash

printf "\n#######################################################\n"
echo have you confirmed that directory is ready for release?
echo make docs
echo make coverage
echo make dist
echo git commit ...

printf "#######################################################\n\n"
echo if you are giving an explicit version have you run bumpversion?
printf "REMOVE 'dev' from version tag (find/replace)\n"
printf "#######################################################\n\n"
echo -n "proceed to release (y/n)? "
read -r answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    clean=$(git status --porcelain --untracked-files=no | wc -l)
    if [ "$clean" -lt 1 ]; then
        if [[ $# == 1 && ($1 == "patch" || $1 == "minor" || $1 == "major") ]]; then
            git checkout release
            git merge master
            bumpversion --tag --commit "$1"
            make dist
            make coverall
            twine upload dist/*.tar.gz
            git push
            git checkout master
            git merge release
            git push
            printf "\n#######################################################\n"
            printf "check that remote builds are successful, then push tags\n"
            printf "if you want this version archived, make sure repository is\n"
            printf "enabled: https://zenodo.org/account/settings/github/\n"
            printf "git push origin <tag>\ngit push release <tag>\n"
            printf "\n#######################################################\n"
        elif [[ $# == 1 && $1 == v*.*.* ]]; then
            git checkout release
            git merge master
            git tag -a "$1" -m "tagged for release $1"
            make dist
            make coverall
            twine upload dist/*.tar.gz
            git push
            git checkout master
            git merge release
            git push
            printf "\n#######################################################\n"
            printf "check that remote builds are successful, then push tags\n"
            printf "if you want this version archived, make sure repository is\n"
            printf "enabled: https://zenodo.org/account/settings/github/\n"
            printf "git push origin <tag>\ngit push release <tag>\n"
            printf "\n#######################################################\n"
        else
            echo usage: ./release.sh "[patch/minor/major]"
            echo usage: ./release.sh "vX.X.X (assumes bumpversion has already been run and vX.X.X matches"
        fi
    else
        echo working directory is not clean!
        git status --porcelain --untracked-files=no
    fi
else
    echo aborted
fi


