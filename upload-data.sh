#!/usr/bin/env bash
echo "Compress files before sending..."
ssh anioche@avakas.mcia.univ-bordeaux.fr
cd aurelien/EcoSimulation-master
tar -cf data.tar data
logout
echo "Files ready to be sent."
scp -r anioche@avakas.mcia.univ-bordeaux.fr:/home/anioche/aurelien/EcoSimulation-master/data.tar /Users/M-E4-ANIOCHE/Desktop/EcoSimulation-data.tar
echo "Decompress files"
cd /Users/M-E4-ANIOCHE/Desktop
tar -xf EcoSimulation-data.tar
echo "Done!"