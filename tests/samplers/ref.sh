oconv box.rad sky176.rad > sky176.oct
oconv box.rad sky174.rad > sky174.oct

vwrays -ff -x 512 -y 512 -vf view.vf | rtrace -ab 1 -ad 10000 -lw 1e-5 -x 512 -y 512 -ld- -ffc -n 12 sky176.oct > sky176.hdr
vwrays -ff -x 512 -y 512 -vf view.vf | rtrace -ab 1 -ad 10000 -lw 1e-5 -x 512 -y 512 -ld- -ffc -n 12 sky174.oct > sky174.hdr

cat sky176.hdr | getinfo -a "VIEW= -vta -vv 180 -vh 180" > temp.hdr
mv temp.hdr sky176.hdr

cat sky174.hdr | getinfo -a "VIEW= -vta -vv 180 -vh 180" > temp.hdr
mv temp.hdr sky174.hdr

oconv box.rad sun176.rad > sun176.oct
oconv box.rad sun174.rad > sun174.oct

vwrays -ff -x 2048 -y 2048 -vf view.vf | rtrace -ab 0 -x 2048 -y 2048 -ld- -ffc sun176.oct | getinfo -a "VIEW= -vta -vv 180 -vh 180" | pfilt -1 -e 1 -x 512 -y 512 -m .25 -r .6 > sun176.hdr
vwrays -ff -x 512 -y 512 -vf view.vf | rtrace -ab 0 -x 512 -y 512 -ld- -ffc sun174.oct | getinfo -a "VIEW= -vta -vv 180 -vh 180" > sun174.hdr

rm *.oct