#!/bin/bash

BACK=$PWD

WORKDIR=/mnt/data5

slides=("slide1" "slide2")
slide_alts=("A" "B")
areas=("A1" "B1" "C1" "D1")
area_alts=("1" "2" "3" "4")

for i in ${!slides[@]};
do
  SLIDE=${slides[$i]}

  cd $WORKDIR/output/count/gi-infection/$SLIDE
  for j in ${!areas[@]};
  do
    AREA=${areas[$j]}
    SAMPLE=${slide_alts[$i]}${area_alts[$j]}

#    echo $SLIDE
#    echo $AREA
#    echo $SAMPLE

    spaceranger count \
      --id=$AREA \
      --transcriptome=$WORKDIR/data/ref-GRCh38-2020-A \
      --probe-set=$WORKDIR/spaceranger/probe_sets/Visium_Human_Transcriptome_Probe_Set_v1.0_GRCh38-2020-A.csv \
      --fastqs=$WORKDIR/data/gi-infection/$SLIDE/$AREA/fastqs \
      --sample=AK-CW-10X-8S-$SAMPLE \
      --image=$WORKDIR/data/gi-infection/$SLIDE/$AREA/$AREA.tif \
      --unknown-slide \
      --reorient-images \
      --localcores=24 \
      --localmem=48

    #  --slide V10L13-021 \
    #  --area B1 \
  done
done

cd $BACK
