#!/bin/bash

echo "gold eval"

rm -r /tmp/golddir/ /tmp/evaldir/
mkdir /tmp/golddir/ /tmp/evaldir/
cp "$1" /tmp/golddir/1.dag
cp "$2" /tmp/evaldir/1.dag

python3 poleval-eval.py /tmp/evaldir/ /tmp/golddir/
