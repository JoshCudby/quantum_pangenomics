#!/usr/bin/perl -w

# Turn a GFA + GAF alignment path string ">node>node<node..." into a
# sequence by copying the elements out of the GFA and stitching them
# together.

# Usage: path2seq.pl in.gfa path

# Parse GFA; minimally
open(my $gfa, "<", shift(@ARGV)) || die;
while (<$gfa>) {
    chomp($_);
    my @F = split(/\s+/, $_);
    next unless scalar(@F) && $F[0] eq "S"; # skip other fields for now
    $gfa{$F[1]}{seq} = uc($F[2]);
}


# Parse the path
while (<>) {
    chomp();
    # Skip lines that don't contain path strings (i.e., '>' or '<')
    next unless m/[<>]/;
    my $seq="";

    foreach (m/[<>][^<>]*/g) {
	my ($dir,$node) = (m/(.)(.*)/);

    # Check if the node exists in the GFA
    if (!exists $gfa{$node}) {
        next;
    }
	my $gseq = $gfa{$node}{seq};
	if ($dir eq "<") {
	    $gseq =~ tr/ACGT/TGCA/;
	    $gseq = reverse($gseq);
	}
	$seq .= $gseq;
    }
    print "path";
    print "$seq\n";
}