#!/usr/bin/env python

import km
import numpy as np
from collections import defaultdict
import argparse

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
    network = {}
    mapper = km.KeplerMapper(verbose=0)
    data = np.zeros((3,3))
    projected_data = mapper.fit_transform( data, projection="sum", scaler=None )

    nodes = defaultdict(list)
    links = defaultdict(list)
    custom = defaultdict(list)

    dat = f.readline()
    lens = f.readline()
    color = f.readline();
    param = [float(i) for i in f.readline().split(" ")]

    nums = [int(i) for i in f.readline().split(" ")]
    num_nodes = nums[0]
    num_edges = nums[1]

    for i in range(0,num_nodes):
        point = [float(j) for j in f.readline().split(" ")]
        nodes[  str(int(point[0]))  ] = [  int(point[0]), point[1], int(point[2])  ]
        links[  str(int(point[0]))  ] = []
        custom[  int(point[0])  ] = point[1]

    m = min([custom[i] for i in range(0,num_nodes)])
    M = max([custom[i] for i in range(0,num_nodes)])

    for i in range(0,num_edges):
        edge = [int(j) for j in f.readline().split(" ")]
        links[  str(edge[0])  ].append(  str(edge[1])  )
        links[  str(edge[1])  ].append(  str(edge[0])  )

    network["nodes"] = nodes
    network["links"] = links
    network["meta"] = lens

    html_output_filename = args.file.rsplit('.', 1)[0] + '.html'
    mapper.visualize(network, color_function=color, path_html=html_output_filename, title=dat,
    graph_link_distance=30, graph_gravity=0.1, graph_charge=-120, custom_tooltips=custom, width_html=0,
    height_html=0, show_tooltips=True, show_title=True, show_meta=True, res=param[0],gain=param[1], minimum=m,maximum=M)
    message = repr(html_output_filename) + " is generated. You can now use your favorite web browser to visualize it."
    print(message)


    f.close()
