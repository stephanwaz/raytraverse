raytraverse -c run.cfg debug sky sunrun
raytraverse -c run.cfg debug integrate > debug_stats.txt
hdrstats  img-cr 'debug_view*.hdr' | hdrstats corr " - debug_stats.txt" -x_vals '0,1' -y_vals '1,0' --rmsd
