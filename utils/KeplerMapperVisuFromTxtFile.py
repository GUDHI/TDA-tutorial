#!/usr/bin/env python

import argparse
from gudhi.cover_complex import _save_to_html

"""This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Mathieu Carriere

   Copyright (C) 2017 Inria

   Modification(s):
     - YYYY/MM Author: Description of the modification
"""

__author__ = "Mathieu Carriere"
__copyright__ = "Copyright (C) 2017 Inria"
__license__ = "GPL v3"

parser = argparse.ArgumentParser(description='Creates an html Keppler Mapper '
                                 'file to visualize a SC.txt file.',
                                 epilog='Example: '
                                 './KeplerMapperVisuFromTxtFile.py '
                                 '-f ../../data/points/human.off_sc.txt'
                                 '- Constructs an human.off_sc.html file.')
parser.add_argument("-f", "--file", type=str, required=True)

args = parser.parse_args()

with open(args.file, 'r') as f:

    dat = f.readline()
    lens = f.readline()
    color = f.readline();
    param = [float(i) for i in f.readline().split(" ")]
    nums = [int(i) for i in f.readline().split(" ")]
    points = [[float(j) for j in f.readline().split(" ")] for i in range(0, nums[0])]
    edges = [[int(j) for j in f.readline().split(" ")]    for i in range(0, nums[1])]
    html_output_filename = args.file.rsplit('.', 1)[0] + '.html'

    f.close()

_save_to_html(dat, lens, color, param, nums, points, edges, html_output_filename)
