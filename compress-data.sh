#!/usr/bin/env bash
tar cf - ../data -P | pv -s $(du -sb ../data | awk '{print $1}')