#!/usr/bin/env bash

(ls data/test.csv data/train.csv &&
    echo "Everything seems fine") ||
(echo "\n/!\\ CSV files are missing /!\\ \n" &&
    exit 1);