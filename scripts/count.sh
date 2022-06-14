#!/bin/bash

WORKDIR=/mnt/data5

spaceranger count --id=colorectal-cancer-count \
  --transcriptome=$WORKDIR/data/ref-GRCh38-2020-A \
  --fastqs=$WORKDIR/data/colorectal-cancer/fastqs \
  --sample=Visium_FFPE_Human_Intestinal_Cancer \
  --image=$WORKDIR/data/colorectal-cancer/brightfield.jpg \
  --slide V10L13-021 \
  --area B1 \
  --localcores=16 \
  --localmem=64
