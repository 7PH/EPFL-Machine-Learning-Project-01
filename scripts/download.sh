#!/usr/bin/env bash

TEST_URL="http://uploads.benjamin-raymond.pro/2018/10/28/21-40-01-000-test.csv";
TRAIN_URL="http://uploads.benjamin-raymond.pro/2018/10/28/21-40-01-001-train.csv";

wget "$TEST_URL" -O data/test.csv;
wget "$TRAIN_URL" -O data/train.csv;