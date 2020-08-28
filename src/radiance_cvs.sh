# git init
# git config core.sparsecheckout true
# echo src/common/ >> .git/info/sparse-checkout
# echo src/rt/ >> .git/info/sparse-checkout
# echo src/CmakeLists.txt >> .git/info/sparse-checkout
# git remote add -f origin https://github.com/NREL/Radiance.git

cvs -d :pserver:anonymous@radiance-online.org:/home/cvsd/radiance co ray/src/common
cvs -d :pserver:anonymous@radiance-online.org:/home/cvsd/radiance co ray/src/rt
cvs -d :pserver:anonymous@radiance-online.org:/home/cvsd/radiance co ray/License.txt 