#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PROJECTOR=old RUN_GROUP=256-tokens_best ./pipeline/run_all.sh
PROJECTOR=new RUN_GROUP=24-tokens_best  ./pipeline/run_all.sh
