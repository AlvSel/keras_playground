#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

data_path   = "./datasets"
out_file    = "calculator_dataset.csv"
myDelimiter = ";"
n_lines     = 10**6
n_arg	    = 4
min_int     = 0
max_int     = 10

def myFunction(args):
	a,b,c,d = args
	return (a+b)*(c+d)

if __name__ == "__main__":
	lines = []
	for i in range(n_lines):
		arg_list = np.random.randint(min_int, max_int, n_arg)
		arg_list = np.append(arg_list,myFunction(arg_list))
		lines.append(arg_list)
	np.savetxt(data_path+'/'+out_file, lines, fmt='%1.0f',delimiter=myDelimiter)
