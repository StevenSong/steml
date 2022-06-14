#!/bin/bash

BACK=$PWD

WORKDIR=/mnt/data5
SLIDE=slide1
AREA=A1
SAMPLE=A1

cd $WORKDIR/output/count/gi-infection/$SLIDE

spaceranger count \
  --id=$AREA \
  --transcriptome=$WORKDIR/data/ref-GRCh38-2020-A \
  --probe-set=$WORKDIR/spaceranger/probe_sets/Visium_Human_Transcriptome_Probe_Set_v1.0_GRCh38-2020-A.csv \
  --fastqs=$WORKDIR/data/gi-infection/$SLIDE/$AREA/fastqs \
  --sample=AK-CW-10X-8S-$SAMPLE \
  --image=$WORKDIR/data/gi-infection/$SLIDE/$AREA/$AREA.tif \
  --unknown-slide \
  --localcores=16 \
  --localmem=48

#  --slide V10L13-021 \
#  --area B1 \

cd $BACK
