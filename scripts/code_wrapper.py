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
import os
import glob
from pathlib import Path
from matplotlib import pyplot as plt

INDEX_OF_KEYWORD_VALUE = 15
PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/builds/gfortran/PATHSAMPLE"

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
        """Add a keywrod."""
        self.input.update({name: value})

    def comment_input(self, name):
        """Comment out a keyword from pathdata file."""
        self.input.pop(name, None)

class ScanPathsample(object):

    def __init__(self, pathdata, outbase=None):
        self.pathdatafile = pathdata #base input file to modify
        self.parse = ParsedPathsample(pathdata=pathdata)
        self.outbase = 'out' #prefix for pathsample output files
        if outbase is not None:
            self.outbase = outbase
        self.outputs = {} #list of output dictionaries

    def run_NGT_regrouped(self, Gthresh):
        """After each regrouping, calculate kNSS on regrouped minima."""
        #Rename regrouped files to min.A and min.B to pass as input to PATHSAMPLE
        files_to_modify = ['min.A', 'min.B', 'min.data', 'ts.data']
        for f in files_to_modify:
            os.system(f"mv {f} {f}.original")
            os.system(f"mv {f}.regrouped.0.2000000000 {f}")
        #run NGT without regroup on regrouped minima
        scan.parse.comment_input('REGROUPFREE')
        scan.parse.comment_input('DUMPGROUPS')
        scan.parse.write_input(scan.pathdatafile)
        outfile_noregroup = f'out.NGT.kNSS.{Gthresh:.2f}'
        os.system(f"{PATHSAMPLE} > {outfile_noregroup}")
        scan.parse.parse_output(outfile=outfile_noregroup)
        kNSSexact = scan.parse.output['kNSSAB']
        #restore original file names
        for f in files_to_modify:
            os.system(f"mv {f} {f}.regrouped.0.2000000000")
            os.system(f"mv {f}.original {f}")
        return kNSSexact
    
    def scan_param(self, name, values, outputkey=None):
        """Re-run PATHSAMPLE calculations by changing
        values of keyword `name` one at a time."""
        corrected_name = str(name).upper()
        outputvals = []
        kNSSs = []
        for value in values:
            #update input
            self.parse.append_input(name, value)
            #overwrite pathdata file with updated input
            self.parse.write_input(self.pathdatafile)
            #run calculation 
            outfile = f'{self.outbase}.{name}.{value:.2f}'
            os.system(f"{PATHSAMPLE} > {outfile}")
            #parse output
            self.parse.parse_output(outfile=outfile)
            #store the output under the value it was run at
            if outputkey is not None:
                outputvals.append(self.parse.output[outputkey])
            self.outputs[value] = self.parse.output
            kNSSs.append(self.run_NGT_regrouped(value))
            print(f"Computed rate constants for regrouped minima with threshold {value}") 
        if outputkey is not None:
            return outputvals, kNSSs

if __name__=='__main__':
    scan = ScanPathsample('./pathdata', outbase='out.NGT')
    temp = 0.2
    scan.parse.append_input('TEMPERATURE', temp)
    nrgthreshs = np.linspace(0.01, 2.0, 100)
    #np.save('nrgthreshs.npy', nrgthreshs)
    kSSs, kNSSs = scan.scan_param('REGROUPFREE', nrgthreshs.tolist(), outputkey='kSSAB')
    #np.save(f'kSSAB_nrgthreshs_T{temp}.npy', kSSs)
    """ This just reproduces the same as result as kNSS without regrouping
    kNSSvals = []
    for value in nrgthreshs:
        kNSSvals.append(scan.outputs[value]['kNSSAB'])
    np.save(f'kNSSAB_Gthresh_T{temp}.npy', kNSSvals)
    """
    #compare to kNSS calculation without free energy regrouping
    scan.parse.comment_input('REGROUPFREE')
    scan.parse.comment_input('DUMPGROUPS')
    scan.parse.write_input(scan.pathdatafile)
    outfile_noregroup = 'out.NGT.NOREGROUP'
    os.system(f"{PATHSAMPLE} > {outfile_noregroup}")
    scan.parse.parse_output(outfile=outfile_noregroup)
    kNSSexact = scan.parse.output['kNSSAB']
    
    #plot kSSAB, kNSSAB as a function of Gthresh, and the exact kNSS
    fig, ax = plt.subplots()
    plt.plot(nrgthreshs, kSSs, '-o', label='kSSAB', markersize=2)
    plt.plot(nrgthreshs, kNSSs, '-o', label='kNSSAB', markersize=2)
    plt.plot(nrgthreshs, np.tile(kNSSexact, len(nrgthreshs)), '--', label='kNSSexact')
    plt.xlabel(r'$G_{thresh}$')
    plt.ylabel(r'$k_{AB}$')
    plt.legend()
    plt.savefig(f'kAB_Gthresh_T{temp}.png')

    #remove all out* files to keep directory clean
    for f in glob.glob('out*'):
        if Path(f).exists():	
            Path(f).unlink() 
    
