#!/bin/bash

# Specify global variables.
DIR_WITH_REPOS="/home/$(echo $(whoami))/git/snn/"
CONDA_ENVIRONMENT_NAME="snncompare"


#######################################
# Verifies a directory exists, throws error otherwise.
# Local variables:
#  dirpath
# Globals:
#  None.
# Arguments:
#  Relative folderpath of folder whose existance is verified.
# Returns:
#  0 If folder was found.
#  31 If the folder was not found.
# Outputs:
#  Nothing
#######################################
manual_assert_dir_exists() {
	local dirpath="$1"
	if [ ! -d "$dirpath" ]; then
		echo "The dir: $dirpath does not exist, even though one would expect it does."
		exit 31
	fi
}


#######################################
# 
# Local variables:
#  
# Globals:
#  
# Arguments:
#  
# Returns:
#  0 If function was evaluated succesfull.
# Outputs:
#  
# TODO(a-t-0):
#######################################
# Source: https://stackoverflow.com/questions/70597896/check-if-conda-env-exists-and-create-if-not-in-bash
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}


#######################################
# 
# Local variables:
#  
# Globals:
#  
# Arguments:
#  
# Returns:
#  0 If function was evaluated succesfull.
# Outputs:
#  
# TODO(a-t-0):
#######################################
conda_env_exists() {
	local some_envirionment="$1"
	if find_in_conda_env ".*$some_envirionment.*" ; then
   		echo "FOUND"
	else
		echo "NOTFOUND"
	fi
}


build_pip_package() {
    local git_path="$1"
    if [ "$(conda_env_exists $CONDA_ENVIRONMENT_NAME)" == "FOUND" ]; then
		eval "$(conda shell.bash hook)"
		cd "$git_path" && conda deactivate && conda activate snncompare && python3 setup.py sdist bdist_wheel
	else
		echo "Error, conda environment name:$CONDA_ENVIRONMENT_NAME not found."
        exit 5
	fi
}


install_pip_package() {
    local git_path="$1"
    if [ "$(conda_env_exists $CONDA_ENVIRONMENT_NAME)" == "FOUND" ]; then
		eval "$(conda shell.bash hook)"
		cd "$git_path" && conda deactivate && conda activate snncompare && pip install -e .
	else
		echo "Error, conda environment name:$CONDA_ENVIRONMENT_NAME not found."
        exit 5
	fi
}


## declare an array variable
declare -a arr=("snnadaptation" "snnalgorithms" "snnbackends" "snncompare" "snnradiation")

for reponame in "${arr[@]}"
do
   echo "$reponame"
   echo ""
   echo ""
   # Assert the required folders exist.
   manual_assert_dir_exists "$DIR_WITH_REPOS$reponame"
   
   # Build the pip package
   build_pip_package "$DIR_WITH_REPOS$reponame"
   
   # Install package locally
   install_pip_package "$DIR_WITH_REPOS$reponame"
   
   # TODO: if CLI arg is passed, upload the pip packages with upgraded version.
done