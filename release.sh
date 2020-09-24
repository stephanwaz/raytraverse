#!/bin/bash
clean=$(git status --porcelain --untracked-files=no | wc -l)
if [ \( $# -eq 1 \) -a \( $clean -lt 1 \) ]
then
	git checkout release
	git merge master
	bumpversion --tag --commit patch
	git push --follow-tags
	git checkout master
	git merge release
	git push
	make dist
	twine upload dist/*.tar.gz
else
	echo usage: ./bump.sh [patch/minor/major/nobump]
fi