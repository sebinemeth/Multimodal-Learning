#!/bin/bash

DATA_PATH="/Users/sebinemeth/Downloads/nvGesture_v1.7z/Video_data/"

for CLS in "$DATA_PATH""class_"*
do
  for SEQ in "$CLS""/subject"*
  do
    echo -n "$SEQ"
    for MOD in {color,depth}
    do
      echo -n "   $MOD"
      rm -r "$SEQ"/sk_"$MOD"_all
      mkdir -p "$SEQ"/sk_"$MOD"_all
      ffmpeg -hide_banner -loglevel error -i "$SEQ"/sk_"$MOD".avi "$SEQ"/sk_"$MOD"_all/%05d.jpg
    done
    echo "   done"
  done
done