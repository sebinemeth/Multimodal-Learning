#!/bin/bash

DATA_PATH="/Users/sebinemeth/Downloads/nvGesture_v1.7z/Video_data/"

for CLS in "$DATA_PATH""class_"*
do
  for SEQ in "$CLS""/subject"*
  do
    for MOD in {color,depth}
    do
      echo $MOD
      rm -r "$SEQ"/sk_"$MOD"_all
      mkdir -p "$SEQ"/sk_"$MOD"_all
      ffmpeg -i "$SEQ"/sk_"$MOD".avi "$SEQ"/sk_"$MOD"_all/%05d.jpg
      echo "$SEQ"
    done
  done
done