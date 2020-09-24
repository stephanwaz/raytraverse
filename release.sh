#!/bin/bash
clean=$(git status --porcelain --untracked-files=no | wc -l)
if [ \( $# -eq 1 \) -a \( "$clean" -lt 1 \) ]
then
	git checkout release
	git merge master
	bumpversion --tag --commit "$1"
	make dist
	twine upload dist/*.tar.gz
	git push --follow-tags
	git checkout master
	git merge release
	git push
else
	echo usage: ./bump.sh [patch/minor/major]
fi
