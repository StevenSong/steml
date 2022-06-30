#!/usr/bin/env python

from steml.recipes import slice, label

name = 'lymphocyte'
conditions = [
    [('SDC1', False, 2),  ('PTPRC', False, 0)],  # plasma cells
    [('MS4A1', False, 0), ('PTPRC', False, 0)],  # B cells
    [('CD3E', False, 0),  ('PTPRC', False, 0)],  # T cells
]

# image_dir = '/mnt/data5/data/gi-infection'
# count_dir = '/mnt/data5/output/count/gi-infection'
# root_dir = '/mnt/data5/output/tiles/gi-infection'
#
# for slide in ['H.pylori', 'C.diff']:
#     for section in ['A1', 'B1', 'C1', 'D1']:
#         output_dir = f'{root_dir}/{slide}/{section}'
#         slice(
#             image=f'{image_dir}/{slide}/{section}/{section}.tif',
#             tissue_positions=f'{count_dir}/{slide}/{section}/outs/spatial/tissue_positions_list.csv',
#             output_dir=output_dir,
#         )
#         label(
#             feature_barcode_matrix=f'{count_dir}/{slide}/{section}/outs/filtered_feature_bc_matrix',
#             conditions=conditions,
#             name=name,
#             output_dir=output_dir,
#         )


image_dir = '/mnt/data5/data/gi-cancer'
count_dir = '/mnt/data5/output/count/gi-cancer'
output_dir = '/mnt/data5/output/tiles/gi-cancer'

slice(
    image=f'{image_dir}/brightfield.jpg',
    tissue_positions=f'{count_dir}/outs/spatial/tissue_positions_list.csv',
    output_dir=output_dir,
)
label(
    feature_barcode_matrix=f'{count_dir}/outs/filtered_feature_bc_matrix',
    conditions=conditions,
    name=name,
    output_dir=output_dir,
)
