SECONDS=0
# echo running clear glass...
# time raytraverse -c scene/01_glz.cfg scene skyrun directskyrun sunrun evaluate pull
# echo running EC glass...
# time raytraverse -c scene/02_ecg.cfg scene skyrun directskyrun sunrun evaluate pull
# echo running rollershade...
# time raytraverse -c scene/03_shd.cfg scene skyrun directskyrun sunrun evaluate pull
# echo running trans...
# time raytraverse -c scene/04_trn.cfg scene skyrun directskyrun sunrun evaluate pull
# echo running glass reflection...
# time raytraverse -c scene/05_ngl.cfg scene skyrun directskyrun sunrun evaluate pull
# echo running metal reflection...
# time raytraverse -c scene/06_nmt.cfg scene skyrun directskyrun sunrun evaluate pull
duration=$SECONDS


# mkdir results
# mv 01_glz_point_-0.60_01.00_01.20_00.00_-1.00_00.00_metric.txt results/01_glz_v2_ryt.tsv
# mv 01_glz_point_00.60_01.00_01.20_01.00_00.00_00.00_metric.txt results/01_glz_v1_ryt.tsv
# mv 01_glz_point_00.60_04.60_01.20_00.00_-1.00_00.00_metric.txt results/01_glz_v3_ryt.tsv
# mv 02_ecg_point_-0.60_01.00_01.20_00.00_-1.00_00.00_metric.txt results/02_ecg_v2_ryt.tsv
# mv 02_ecg_point_00.60_01.00_01.20_01.00_00.00_00.00_metric.txt results/02_ecg_v1_ryt.tsv
# mv 02_ecg_point_00.60_04.60_01.20_00.00_-1.00_00.00_metric.txt results/02_ecg_v3_ryt.tsv
# mv 03_shd_point_-0.60_01.00_01.20_00.00_-1.00_00.00_metric.txt results/03_shd_v2_ryt.tsv
# mv 03_shd_point_00.60_01.00_01.20_01.00_00.00_00.00_metric.txt results/03_shd_v1_ryt.tsv
# mv 03_shd_point_00.60_04.60_01.20_00.00_-1.00_00.00_metric.txt results/03_shd_v3_ryt.tsv
# mv 04_trn_point_-0.60_01.00_01.20_00.00_-1.00_00.00_metric.txt results/04_trn_v2_ryt.tsv
# mv 04_trn_point_00.60_01.00_01.20_01.00_00.00_00.00_metric.txt results/04_trn_v1_ryt.tsv
# mv 04_trn_point_00.60_04.60_01.20_00.00_-1.00_00.00_metric.txt results/04_trn_v3_ryt.tsv
# mv 05_ngl_point_-0.60_-1.00_01.20_-1.00_00.00_00.00_metric.txt results/05_ngl_v1_ryt.tsv
# mv 05_ngl_point_-0.60_-4.60_01.20_00.00_01.00_00.00_metric.txt results/05_ngl_v3_ryt.tsv
# mv 05_ngl_point_00.60_-1.00_01.20_00.00_01.00_00.00_metric.txt results/05_ngl_v2_ryt.tsv
# mv 06_nmt_point_-0.60_-1.00_01.20_-1.00_00.00_00.00_metric.txt results/06_nmt_v1_ryt.tsv
# mv 06_nmt_point_-0.60_-4.60_01.20_00.00_01.00_00.00_metric.txt results/06_nmt_v3_ryt.tsv
# mv 06_nmt_point_00.60_-1.00_01.20_00.00_01.00_00.00_metric.txt results/06_nmt_v2_ryt.tsv


filen=$(date "+%Y_%m_%d-%H-%M-%S")

date > "$filen"_report.txt

if [[ $OSTYPE == 'darwin'* ]]; then
	system_profiler SPHardwareDataType >> "$filen"_report.txt
else
	lscpu
	fi

./check.py 01_glz 02_ecg 03_shd 04_trn 05_ngl 06_nmt >> "$filen"_report.txt
perpoint=$(rcalc -n -e '$1='$duration'/18')
printf "\nsimulation time: $duration (per point: $perpoint)\n\n" >> "$filen"_report.txt
#
printf "\n"
cat "$filen"_report.txt


