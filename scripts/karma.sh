#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m'

ensure_command_exists() {
    command -v $1 >/dev/null 2>&1 || { echo >&2 "$RED I require $1 but it's not installed. $NC"; exit 1; }
}

ensure_file_exit() {
    ls $1 || (echo "$RED/!\\ File $1 is missing /!\\ $NC" && exit 1);
}

echo "\nVerifying that csv files are present..";
ensure_file_exit "data/test.csv";
ensure_file_exit "data/train.csv";

echo "\nVerifying that python linter is installed..";
ensure_command_exists "pylint";

echo "\nEverything seems OK";