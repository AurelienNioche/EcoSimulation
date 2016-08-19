#!/usr/bin/env bash
# On Mac OSX
tar cf - /folder-with-big-files -P | pv -s $(($(du -sk /folder-with-big-files | awk '{print $1}') * 1024))