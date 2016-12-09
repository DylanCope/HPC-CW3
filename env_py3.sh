# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions
module add shared default-environment
module add languages/python-3.3.2
module add tools/git-1.8.4.2
module add openmpi/gcc/64/1.6.5
module add tools/gnu_builds/tau-2.23.1-openmpi
module add tools/gnu_builds/tau-2.23.1-openmp
# module load tools/gnu_builds/tau-2.23.1-openmpi
module add languages/intel-compiler-16
alias watchme="qstat -u dc14770"
alias watchqueue="qstat | grep teaching"
