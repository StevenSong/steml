#!/bin/bash

spaceranger count --id=gi-demo \
  --transcriptome=/home/ssong/work/refdata-gex-GRCh38-2020-A \
  --fastqs=/home/ssong/work/data/demo/fastqs \
  --sample=Visium_FFPE_Human_Intestinal_Cancer \
  --image=/home/ssong/work/data/demo/Visium_FFPE_Human_Intestinal_Cancer_image.jpg \
  --slide V10L13-021 \
  --area B1 \
  --localcores=16 \
  --localmem=64

