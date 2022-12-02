#!/bin/bash

## declare the list of github repositories in an array.
declare -a arr=("snnadaptation" "snnalgorithms" "snnbackends" "snncompare" "snnradiation")
# Specify global variables.
DIR_WITH_REPOS="/home/$(echo $(whoami))/git/snn/"
CONDA_ENVIRONMENT_NAME="snncompare"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rebuild) rebuild=1; ;;
        -p|--precommit) precommit=1 ;;
		-c|--commitpush)
			commitpush=1
			COMMIT_MESSAGE="$2"
      		shift # past argument
      		shift # past value
      		;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

#######################################
# Verifies a directory exists, throws error otherwise.
# Local variables:
#  dirpath
# Globals:
#  None.
# Arguments:
#  Relative folderpath of folder whose existence is verified.
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
#  0 If function was evaluated successful.
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
#  0 If function was evaluated successful.
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


run_precommit() {
    local git_path="$1"
    if [ "$(conda_env_exists $CONDA_ENVIRONMENT_NAME)" == "FOUND" ]; then
		eval "$(conda shell.bash hook)"
		cd "$git_path" && conda deactivate && conda activate snncompare && git add *
		cd "$git_path" && conda deactivate && conda activate snncompare && git add .* && pre-commit run --all-files
		cd "$git_path" && conda deactivate && conda activate snncompare && pre-commit install
		cd "$git_path" && conda deactivate && conda activate snncompare && pre-commit run --all-files
	else
		echo "Error, conda environment name:$CONDA_ENVIRONMENT_NAME not found."
        exit 5
	fi
}


commit_and_push() {
    local git_path="$1"
	local message="$2"
    if [ "$(conda_env_exists $CONDA_ENVIRONMENT_NAME)" == "FOUND" ]; then
		eval "$(conda shell.bash hook)"
		cd "$git_path" && conda deactivate && conda activate snncompare && git add *
		cd "$git_path" && conda deactivate && conda activate snncompare && git add -A
		cd "$git_path" && conda deactivate && conda activate snncompare && git commit -m "$message"
		cd "$git_path" && conda deactivate && conda activate snncompare && git push
	else
		echo "Error, conda environment name:$CONDA_ENVIRONMENT_NAME not found."
        exit 5
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

for reponame in "${arr[@]}"
do
	echo "$reponame"
	echo ""
	echo ""
	# Assert the required folders exist.
	manual_assert_dir_exists "$DIR_WITH_REPOS$reponame"

	if [ "$rebuild" == 1 ]; then
		# Build the pip package
   		build_pip_package "$DIR_WITH_REPOS$reponame"

   		# Install package locally
   		install_pip_package "$DIR_WITH_REPOS$reponame"
	fi

	if [ "$precommit" == 1 ]; then
		run_precommit "$DIR_WITH_REPOS$reponame"
	fi

	if [ "$commitpush" == 1 ]; then
		commit_and_push "$DIR_WITH_REPOS$reponame" "$COMMIT_MESSAGE"
	fi

   	# TODO: if CLI arg is passed, upload the pip packages with upgraded version.
done
