""" Script to iteratively run PATHSAMPLE and analyze its output.

Workflow:
- read in initial pathdata file, append / change parameters as desired
- use a different class to run calculations and name output files
- use parser class again to then extract output
- so one ParsedPathsample object per calculation? 

TODO: potentially parse output separately?

Deepti Kannan 2019."""

import re
import numpy as np
import sys
from pathlib import Path

INDEX_OF_KEYWORD_VALUE = 15

class ParsedPathsample(object):
    
    def __init__(self, outfile=None, pathdata=None):
        self.output = {} #dictionary of output (i.e. temperature, rates)
        self.input = {} #dictionary of keywords
        if outfile is not None:
            self.parse_output(outfile)
        if pathdata is not None:
            self.parse_input(pathdata)

    def parse_output(self, outfile):
        """Searches for output of various subroutines of pathsample,
        including NGT, REGROUPFREE, and DIJKSTRA.

        TODO: test if robust to all pathsample output files
        TODO: make generalizable for subroutines other than NGT
        """
        with open(outfile) as f:
            for line in f:
                if not line:
                    continue
                words = line.split()
                if len(words) > 1:
                    if words[0] == 'Temperature=':
                        #parse and spit out self.temperature
                        self.output['temperature'] = float(words[1])
                    if words[0] == 'NGT>' and words[1]=='kSS':
                        self.output['kSSAB'] = float(words[3])
                        self.output['kSSBA'] = float(words[6])
                    if words[0] == 'NGT>' and words[1]=='kNSS(A<-B)=':
                        print(words)
                        self.output['kNSSAB'] = float(words[2])
                        self.output['kNSSBA'] = float(words[4])

    def parse_input(self, pathdata):
        """Store keywords in pathdata as a dictionary.
        Note: stores all values as strings
        """
        name_value_re = re.compile('([_A-Za-z0-9][_A-Za-z0-9]*)\s*(.*)\s*')

        with open(pathdata) as f:
            for line in f:
                if not line or line[0]=='!':
                    continue
                match = name_value_re.match(line)
                if match is None:
                    continue
                name, value = match.groups()
                self.input.update({name: value})

    def write_input(self, pathdatafile):
        """Writes out a valid pathdata file based on keywords
        in self.input"""
        with open(pathdatafile, 'w') as f:
            #first 3 lines are garbage
            f.write('! PATHSAMPLE input file generated from\n')
            f.write('! ParsedPathsample class\n\n')
            for name in self.input:
                name_length = len(name)
                #the value of the keyword begins on the 15th index of the line
                #add the appropriate number of spaces before the value
                numspaces = INDEX_OF_KEYWORD_VALUE - name_length
                f.write(str(name).upper() + ' '*numspaces + str(self.input[name]) + '\n')

    def append_input(self, name, value):
        self.input.update({name: value})

class ScanPathsample(object):

    def __init__(self, pathdata, outbase=None):
        self.pathdatafile = pathdata #base input file to modify
        self.parse = ParsedPathsample(pathdata=pathdata)
        self.outbase = 'out.' #prefix for pathsample output files
        if outbase is not None:
            self.outbase = outbase
        self.outputs = {} #list of output dictionaries

    def scan_param(self, name, values, outputkey=None):
        """Re-run PATHSAMPLE calculations by changing
        values of keyword `name` one at a time."""
        corrected_name = str(name).upper()
        outputvals = []
        for value in values:
            #update input
            self.parse.append_input(name, value)
            #overwrite pathdata file with updated input
            self.parse.write_input(self.pathdatafile)
            #run calculation 
            outfile = f'{self.outbase}.{name}.{value}'
            sys.stdout(f'PATHSAMPLE > {outfile}')
            #parse output
            self.parse.parse_output(outfile=outfile)
            #store the output under the value it was run at
            if outputkey is not None:
                outputvals.append(self.parse.output[outputkey])
            self.outputs[value] = self.parse.output
        
        if output key is not None:
            return values, outputvals
    

