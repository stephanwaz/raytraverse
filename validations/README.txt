Each folder contains two shell scripts:

	- run.sh
	- clean.sh

run.sh will produce a report to benchmark performance on the given hardware. To
explore other settings/simulation types, change the settings in all of the .cfg
files in the scene/ subdirectory which contains all of the dependencies. Some 
directories have relative file references to depend on the scene/ files in
another directory. Note that these runs use a reduce -ab to match the reference
simulations, this is not recommended for typical use as it only results in a 
slight time savings given the way raytraverse uses radiance.