#!/bin/bash
clean=$(git status --porcelain --untracked-files=no | wc -l)
nobump=nobump
if [ \( $# -eq 1 \) -a \( $clean -lt 1 \) ]
then
	if [ "$1" == "$nobump" ]
	then
		commitmessage='small version error'
	else
		eval "$(bumpversion --list "$1")"
		commitmessage='Bump version: '"$current_version"' → '"$new_version"''
	fi
	echo $commitmessage
	read -p "proceed to build docs? " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		mate -w history.rst
		make docs
		git add docs/
	fi

	read -p "proceed with commit? " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		git commit -a -m "$commitmessage"
		if [ "$1" != "$nobump" ]
		then
			git tag v$new_version
			git push --tags
		fi
	fi
	read -p "build? " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		make release
	fi

	make log

	echo $message
else
	echo usage: ./bump.sh [patch/minor/major/nobump]
fi
