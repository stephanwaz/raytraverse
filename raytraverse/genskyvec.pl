#!/usr/bin/perl -w
# RCSid $Id: genskyvec.pl,v 2.11 2018/06/05 01:25:11 greg Exp $
#
# Generate Reinhart vector for a given sky description
#
#	G. Ward
#
use strict;
my $windoz = ($^O eq "MSWin32" or $^O eq "MSWin64");
my @skycolor = (0.960, 1.004, 1.118);
my $mf = 4;
my $dosky = 1;
my $dosun = 1;
my $dofive = 0;
#SW edit
my $onepatch = 0;
my $brightness = 0;
my $ncomp = 3;
my $shirleychiu = 0;
my $headout = 1;
my @origARGV = @ARGV;
while ($#ARGV >= 0) {
	if ("$ARGV[0]" eq "-c") {
		@skycolor = @ARGV[1..3];
		shift @ARGV for (1..3);
	} elsif ("$ARGV[0]" eq "-m") {
		$mf = $ARGV[1];
		shift @ARGV;
	} elsif ("$ARGV[0]" eq "-d") {
		$dosky = 0;
    } elsif ("$ARGV[0]" eq "-s") {
        $dosun = 0;
	} elsif ("$ARGV[0]" eq "-5") {
		$dofive = 1;
	# SW edit
    } elsif ("$ARGV[0]" eq "-1") {
        $onepatch = 1;
    } elsif ("$ARGV[0]" eq "-b") {
        $brightness = 1;
	} elsif ("$ARGV[0]" eq "-sc") {
		$shirleychiu = 1;
	} elsif ("$ARGV[0]" eq "-h") {
		$headout = 0;
	} else {
		die "Unexpected command-line argument: $ARGV[0]\n";
	}
	shift @ARGV;
}
if ($brightness) {
    @skycolor = (1.0, 1.0, 1.0);
    $ncomp = 1;
}
# Load sky description into line array, separating sun if one
my @skydesc;
my $lightline;
my @sunval;
my $sunline;
my $skyOK = 0;
my $srcmod;	# putting this inside loop breaks code(?!)
while (<>) {
	push @skydesc, $_;
	if (/^\w+\s+light\s+/) {
		s/\s*$//; s/^.*\s//;
		$srcmod = $_;
		$lightline = $#skydesc;
	} elsif (defined($srcmod) && /^($srcmod)\s+source\s/) {
		@sunval = split(' ', $skydesc[$lightline + 3]);
        @sunval = @sunval[1..$ncomp];
		$sunline = $#skydesc;
	} elsif (/\sskyfunc\s*$/) {
		$skyOK = 1;
	}
}
die "Bad sky description!\n" if (! $skyOK);
# Strip out the solar source if present
my @sundir;
if (defined $sunline) {
	@sundir = split(' ', $skydesc[$sunline + 3]);
	shift @sundir;
	undef @sundir if ($sundir[2] <= 0);
	splice(@skydesc, $sunline, 5);
}
# SW edit
my $rhcal;
if ($shirleychiu) {
	$rhcal =
		'mod(n,d) : n - floor(n/d)*d;' .
		'sq(x) : x*x;' .
		'abs(x) : if( x, x, -x );' .
		'FTINY : 1e-7;' .
		'PI : 3.14159265358979323846;' .
		'x1 = .5;' .
		'x2 = .5;' .
		'Rmax : sq(MF);' .
		'Romega : 2*PI/Rmax;' .
		'bin = Rbin - 1;' .
		'U = if(bin+FTINY, (bin - mod(bin, MF)) / Rmax + x1/MF, 1+x1);' .
		'V = if(bin+FTINY, mod(bin, MF)/MF + x2/MF, x2);' .
		'n = if(U - 1, -1, 1);' .
		'ur = if(U - 1, U - 1, U);' .
		'a = 2 * ur - 1;' .
		'b = 2 * V - 1;' .
		'conda = sq(a) - sq(b);' .
		'condb = abs(b) - FTINY;' .
		'r = if(conda, a, if(condb, b, 0));' .
		'phi = if(conda, b/(2*a), if(condb, 1 - a/(2*b), 0)) * PI/2;' .
		'sphterm = r * sqrt(2 - sq(r));' .
		'Dx = n * cos(phi)*sphterm;' .
		'Dy = sin(phi)*sphterm;' .
		'Dz = n * (1 - sq(r));' ;
}
else {
	# Reinhart sky sample generator
	$rhcal = 'DEGREE : PI/180;' .
		'x1 = .5; x2 = .5;' .
		'alpha : 90/(MF*7 + .5);' .
		'tnaz(r) : select(r, 30, 30, 24, 24, 18, 12, 6);' .
		'rnaz(r) : if(r-(7*MF-.5), 1, MF*tnaz(floor((r+.5)/MF) + 1));' .
		'raccum(r) : if(r-.5, rnaz(r-1) + raccum(r-1), 0);' .
		'RowMax : 7*MF + 1;' .
		'Rmax : raccum(RowMax);' .
		'Rfindrow(r, rem) : if(rem-rnaz(r)-.5, Rfindrow(r+1, rem-rnaz(r)), r);' .
		'Rrow = if(Rbin-(Rmax-.5), RowMax-1, Rfindrow(0, Rbin));' .
		'Rcol = Rbin - raccum(Rrow) - 1;' .
		'Razi_width = 2*PI / rnaz(Rrow);' .
		'RAH : alpha*DEGREE;' .
		'Razi = if(Rbin-.5, (Rcol + x2 - .5)*Razi_width, 2*PI*x2);' .
		'Ralt = if(Rbin-.5, (Rrow + x1)*RAH, asin(-x1));' .
		'Romega = if(.5-Rbin, 2*PI, if(Rmax-.5-Rbin, ' .
		'	Razi_width*(sin(RAH*(Rrow+1)) - sin(RAH*Rrow)),' .
		'	2*PI*(1 - cos(RAH/2)) ) );' .
		'cos_ralt = cos(Ralt);' .
		'Dx = sin(Razi)*cos_ralt;' .
		'Dy = cos(Razi)*cos_ralt;' .
		'Dz = sin(Ralt);' ;
}
my ($nbins, $octree, $tregcommand, $suncmd);
if ($windoz) {
	$nbins = `rcalc -n -e MF:$mf -e \"$rhcal\" -e \"\$1=Rmax+1\"`;
	chomp $nbins;
	$octree = "gtv$$.oct";
	$tregcommand = "cnt $nbins 16 | rcalc -e MF:$mf -e \"$rhcal\" " .
		q{-e "Rbin=$1;x1=rand(recno*.37-5.3);x2=rand(recno*-1.47+.86)" } .
		q{-e "$1=0;$2=0;$3=0;$4=Dx;$5=Dy;$6=Dz" } .
		"| rtrace -h -ab 0 -w $octree | total -16 -m";
	if (@sundir) {
		$suncmd = "cnt " . ($nbins-1) .
			" | rcalc -e MF:$mf -e \"$rhcal\" -e Rbin=recno " .
			"-e \"dot=Dx*$sundir[0] + Dy*$sundir[1] + Dz*$sundir[2]\" " .
			"-e \"cond=dot-.866\" " .
			q{-e "$1=if(1-dot,acos(dot),0);$2=Romega;$3=recno" };
	}
} else {
	$nbins = `rcalc -n -e MF:$mf -e \'$rhcal\' -e \'\$1=Rmax+1\'`;
	chomp $nbins;
	$octree = "/tmp/gtv$$.oct";
    # $tregcommand = "cnt $nbins | rcalc -of -e MF:$mf -e '$rhcal' " .
	# 	q{-e 'Rbin=$1;' } .
	# 	q{-e '$1=0;$2=0;$3=0;$4=Dx;$5=Dy;$6=Dz' } .
	# 	"| rtrace -h -ff -ab 0 -w $octree | total -if3 -1 -m";
    $tregcommand = "cnt $nbins 16 | rcalc -of -e MF:$mf -e '$rhcal' " .
		q{-e 'Rbin=$1;bn=mod(recno, 16);x1=(bn - mod(bn, 4))/16 + .5/4;x2=mod(bn, 4)/4 + .5/4' } .
		q{-e '$1=0;$2=0;$3=0;$4=Dx;$5=Dy;$6=Dz' } .
		"| rtrace -h -ff -ab 0 -w $octree | total -if3 -16 -m";
	# $tregcommand = "cnt $nbins 16 | rcalc -of -e MF:$mf -e '$rhcal' " .
	# 	q{-e 'Rbin=$1;x1=rand(recno*.37-5.3);x2=rand(recno*-1.47+.86)' } .
	# 	q{-e '$1=0;$2=0;$3=0;$4=Dx;$5=Dy;$6=Dz' } .
	# 	"| rtrace -h -ff -ab 0 -w $octree | total -if3 -16 -m";
	if (@sundir) {
		$suncmd = "cnt " . ($nbins-1) .
			" | rcalc -e MF:$mf -e '$rhcal' -e Rbin=recno " .
			"-e 'dot=Dx*$sundir[0] + Dy*$sundir[1] + Dz*$sundir[2]' " .
			"-e 'cond=dot-.866' " .
			q{-e '$1=if(1-dot,acos(dot),0);$2=Romega;$3=recno' };
	}
}
my $empty = "0\t0\t0\n";
if ($brightness) {
    $tregcommand .= q{ | rcalc -e '$1=$1'};
    $empty = "0\n";
}
my @tregval;
if ($dosky) {
	# Create octree for rtrace
	open OCONV, "| oconv - > $octree";
	print OCONV @skydesc;
	print OCONV "skyfunc glow skyglow 0 0 4 @skycolor 0\n";
	print OCONV "skyglow source sky 0 0 4 0 0 1 360\n";
	close OCONV;
	# Run rtrace and average output for every 16 samples
	@tregval = `$tregcommand`;
	unlink $octree;
} else {
	push @tregval, $empty for (1..$nbins);
}
# Find closest patch(es) to sun and divvy up direct solar contribution
sub numSort1 {
	my @a1 = split("\t", $a);
	my @b1 = split("\t", $b);
	return ($a1[0] <=> $b1[0]);
}
if (@sundir) {
	my @bestdir = `$suncmd`;
	@bestdir = sort numSort1 @bestdir;
    #SW edit
    if ($dosun) {
        if ($onepatch) {
            my ($ang, $dom, $ndx);
            ($ang, $dom, $ndx) = split(' ', $bestdir[0]);
            my $somega = ($sundir[3] / 360) ** 2 * 3.141592654 ** 3;
            my @scolor = split(' ', $tregval[$ndx]);
            for (my $j = 0; $j < $ncomp; $j++) {$scolor[$j] += $sunval[$j] * $somega / $dom;}
            $tregval[$ndx] = join("\t", @scolor) . "\n";
        }
        elsif ($dofive) {
            my ($ang, $dom, $ndx);
            ($ang, $dom, $ndx) = split(' ', $bestdir[0]);
            $tregval[$ndx] = join("\t", @sunval) . "\n";
        }
        else {
            my (@ang, @dom, @ndx);
            my $somega = ($sundir[3] / 360) ** 2 * 3.141592654 ** 3;
            my $wtot = 0;
            for my $i (0 .. 2) {
                ($ang[$i], $dom[$i], $ndx[$i]) = split(' ', $bestdir[$i]);
                $wtot += 1. / ($ang[$i] + .02);
            }
            for my $i (0 .. 2) {
                my $wt = 1. / ($ang[$i] + .02) / $wtot * $somega / $dom[$i];
                my @scolor = split(' ', $tregval[$ndx[$i]]);
                for (my $j = 0; $j < $ncomp; $j++) {$scolor[$j] += $wt * $sunval[$j];}
                $tregval[$ndx[$i]] = join("\t", @scolor) . "\n";
            }
        }
    }
}
# Output header if requested
if ($headout) {
	print "#?RADIANCE\n";
	print "genskyvec @origARGV\n";
	print "NROWS=", $#tregval+1, "\n";
	print "NCOLS=1\nNCOMP=", $ncomp,"\n";
	print "FORMAT=ascii\n";
	print "\n";
}
# Output our final vector
print @tregval;
