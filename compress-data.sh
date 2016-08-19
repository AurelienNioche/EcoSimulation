#!/usr/bin/env bash
# On Linux
tar cf - ../data -P | pv -s $(du -sb ../data | awk '{print $1}')