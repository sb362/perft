#!/bin/sh
g++ -std=c++17 -Ifmt/include -m64 -mbmi2 -msse4 -g perft.cc -o perft
