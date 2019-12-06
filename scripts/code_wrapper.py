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
import pandas as pd
from pathlib import Path

MAXINA = 5
MAXINB = 395
INDEX_OF_KEYWORD_VALUE = 15
PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"

class ParsedPathsample(object):
    
    def __init__(self, pathdata, maxinA=5, maxinB=395, outfile=None):
        self.output = {} #dictionary of output (i.e. temperature, rates)
        self.input = {} #dictionary of PATHSAMPLE keywords
        self.numInA = 0 #number of minima in A
        self.numInB = 0 #number of minima in B
        self.minA = [] #IDs of minima in A set
        self.minB = [] #IDs of minima in B set
        self.maxinA = maxinA #maximum allowed number of minima in A set
        self.maxinB = maxinB #maximum allowed number of minima in B set
        #read in numInA, numInB, minA, minB from min.A and min.B files
        self.path = Path(pathdata).parent.absolute()
        self.parse_minA_and_minB(self.path/'min.A', self.path/'min.B')
        if outfile is not None:
            self.parse_output(outfile)
        self.parse_input(pathdata)
    
    def parse_minA_and_minB(self, minA, minB):
        """Read in the number of minima and the minima IDs in the A and B sets
        from the min.A and min.B files. Note, minima IDs in these files
        correspond to line numbers in min.data. However, in this class, we
        subtract 1 from the IDs to correspond to indices in the python data
        strustructure."""
        Aids = []
        with open(minA) as f:
            for line in f:
                Aids = Aids + line.split()
        Aids = np.array(Aids).astype(int)
        maxinA = Aids[0]
        #adjust indices by 1 cuz python is 0-indexed
        Aids = Aids[1:] - 1
        self.minA = Aids
        self.numInA = maxinA
        #and for min.B
        Bids = []
        with open(minB) as f:
            for line in f:
                Bids = Bids + line.split()
        Bids = np.array(Bids).astype(int)
        maxinB = Bids[0]
        Bids = Bids[1:] - 1
        self.minB = Bids
        self.numInB = maxinB

    def sort_A_and_B(self, minA, minB, mindata):
        """Sort the minima in min.A and min.B according to their energies."""
        self.parse_minA_and_minB(minA, minB)
        min_nrgs = np.loadtxt(mindata).astype(float)[:,0]
        minA_nrgs = min_nrgs[self.minA]
        sorted_idAs = np.argsort(minA_nrgs)
        self.minA = self.minA[sorted_idAs]
        print(self.minA)
        minB_nrgs = min_nrgs[self.minB]
        sorted_idBs = np.argsort(minB_nrgs)
        self.minB = self.minB[sorted_idBs]
        print(self.minB)
        self.write_minA_minB(self.path/'min.A', self.path/'min.B')

    def define_A_and_B(self, numInA, numInB, sorted=True, mindata=None):
        """Define an A and B set as a function of the number of minima in A
        and B. """
        if numInA > self.maxinA:
            raise ValueError(f'The maximum allowed number of states in A is {self.maxinA}')
        if numInB > self.maxinB:
            raise ValueError(f'The maximum allowed number of states in B is {self.maxinB}')
        if sorted:
            #min.A and min.B are already sorted by energy
            #just change numInA and numInB
            self.numInA = numInA
            self.numInB = numInB
            return
        if mindata is None:
            mindata = self.path/'min.data'
        #first column of min.data gives free energies
        min_nrgs = np.loadtxt(mindata).astype(float)[:,0]
        #index minima by 0 to correspond to indices of min_nrgs
        minIDs = np.arange(0, len(min_nrgs), 1)
        minA_nrgs = min_nrgs[self.minA]
        minB_nrgs = min_nrgs[self.minB]
        idA = np.argpartition(minA_nrgs, numInA)
        self.minA = self.minA[idA[:numInA]]
        self.numInA = numInA
        idB = np.argpartition(minB_nrgs, numInB)
        self.minB = self.minB[idB[:numInB]]
        self.numInB = numInB
        
    def write_minA_minB(self, minA, minB):
        """Write a min.A and min.B file based on minIDs
        specified in self.minA and self.minB"""
        with open(minA,'w') as f:
            f.write(str(self.numInA)+'\n') #first line is number of minima
            for min in self.minA:
                f.write(str(min+1)+'\n')

        with open(minB,'w') as f:
            f.write(str(self.numInB)+'\n') #first line is number of minima
            for min in self.minB:
                f.write(str(min+1)+'\n')

    def parse_output(self, outfile):
        """Searches for output of various subroutines of pathsample,
        including NGT, REGROUPFREE, and DIJKSTRA.

        TODO: include the true NGT rate constant from disconnect sources
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
                    if words[0] == 'NGT>' and words[1]=='k(A<-B)=':
                        self.output['kAB'] = float(words[2])
                        self.output['kBA'] = float(words[4])

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
            #f.write('! PATHSAMPLE input file generated from\n')
            #f.write('! ParsedPathsample class\n\n')
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
        """Delete a keyword from pathdata file."""
        self.input.pop(name, None)

class ScanPathsample(object):

    def __init__(self, pathdata, outbase=None, suffix=''):
        self.pathdatafile = pathdata #base input file to modify
        self.parse = ParsedPathsample(pathdata=pathdata)
        self.path = Path(pathdata).parent.absolute()
        self.suffix = suffix #suffix of rates_{suffix}.csv file output
        self.outbase = 'out' #prefix for pathsample output files
        if outbase is not None:
            self.outbase = outbase
        self.outputs = {} #list of output dictionaries

    def run_NGT_regrouped(self, Gthresh, temp, direction=None):
        """After each regrouping, calculate kNSS on regrouped minima."""
        #Rename regrouped files to min.A and min.B to pass as input to PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f"mv {f} {f}.original")
            os.system(f"mv {f}.regrouped.{temp:.10f} {f}")
        #run NGT without regroup on regrouped minima
        scan.parse.comment_input('REGROUPFREE')
        scan.parse.comment_input('DUMPGROUPS')
        scan.parse.write_input(scan.pathdatafile)
        outfile_noregroup = self.path/f'out.NGT.kNSS.{Gthresh:.2f}'
        os.system(f"{PATHSAMPLE} > {outfile_noregroup}")
        scan.parse.parse_output(outfile=outfile_noregroup)
        kNSSAB = scan.parse.output['kNSSAB']
        kNSSBA = scan.parse.output['kNSSBA']
        #restore original file names
        for f in files_to_modify:
            os.system(f"mv {f} {f}.regrouped.{temp:.10f}")
            os.system(f"mv {f}.original {f}")
        if direction is None:
            return kNSSAB, kNSSBA
        if direction is 'BA':
            return kNSSBA
        if direction is 'AB':
            return kNSSAB
    
    def run_NGT_exact(self, direction=None):
        #compare to exact kNSS calculation without free energy regrouping
        self.parse.comment_input('REGROUPFREE')
        self.parse.comment_input('DUMPGROUPS')
        self.parse.write_input(scan.pathdatafile)
        outfile_noregroup = self.path/'out.NGT.NOREGROUP'
        os.system(f"{PATHSAMPLE} > {outfile_noregroup}")
        scan.parse.parse_output(outfile=outfile_noregroup)
        kNSSAB = scan.parse.output['kNSSAB']
        kNSSBA = scan.parse.output['kNSSBA']
        if direction is None:
            return kNSSAB, kNSSBA
        if direction is 'BA':
            return kNSSBA
        if direction is 'AB':
            return kNSSAB

    def scan_regroup(self, name, values, temp, outputkey='kSSAB'):
        """Re-run PATHSAMPLE calculations for different `values` of the
        REGROUPFREE threshold. Extract output defined by outputkey and run NGT
        on the regrouped minima to get the SS/NSS rate constants."""
        corrected_name = str(name).upper()
        csv = Path(f'csvs/rates_{self.suffix}.csv')
        dfs = []
        for value in values:
            df = pd.DataFrame()
            #update input
            self.parse.append_input(name, value)
            #overwrite pathdata file with updated input
            self.parse.write_input(self.pathdatafile)
            #run calculation 
            outfile = self.path/f'{self.outbase}.{name}.{value:.2f}'
            os.system(f"{PATHSAMPLE} > {outfile}")
            #parse output
            self.parse.parse_output(outfile=outfile)
            #store the output under the value it was run at
            df[outputkey] = [self.parse.output[outputkey]]
            self.outputs[value] = self.parse.output
            kNSSAB, kNSSBA = self.run_NGT_regrouped(value, temp)
            df['kNSSAB'] = [kNSSAB]
            df['kNSSBA'] = [kNSSBA]
            df['Gthresh'] = [value]
            dfs.append(df)
            print(f"Computed rate constants for regrouped minima with threshold {value}") 
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        bigdf['T'] = self.temp
        kexact_AB, kexact_BA = self.run_NGT_exact()
        bigdf['kNSSexactAB'] = kexact_AB
        bigdf['kNSSexactBA'] = kexact_BA
        #if file exists, append to existing data
        if csv.is_file():
            olddf = pd.read_csv(csv)
            bigdf = olddf.append(bigdf)
        #write updated file to csv
        bigdf.to_csv(csv, index=False)
    
    def scan_temp(self, name, values):
        """Re-run PATHSAMPLE at different temperatures specified by values and
        extract kNSSAB and kNSSBA from the NGT keyword output."""
        corrected_name = str(name).upper()
        csv = Path(f'csvs/rates_{self.suffix}.csv')
        dfs = []
        for value in values:
            df = pd.DataFrame()
            #update input
            self.parse.append_input(name, value)
            #overwrite pathdata file with updated input
            self.parse.write_input(self.pathdatafile)
            #run calculation 
            outfile = self.path/f'{self.outbase}.{name}.{value:.2f}'
            os.system(f"{PATHSAMPLE} > {outfile}")
            #parse output
            self.parse.parse_output(outfile=outfile)
            #store the output under the value it was run at
            self.outputs[value] = self.parse.output
            df['kNSSAB'] = [self.parse.output['kNSSAB']]
            df['kNSSBA'] = [self.parse.output['kNSSBA']]
            df['kSSAB'] = [self.parse.output['kSSAB']]
            df['kSSBA'] = [self.parse.output['kSSBA']]
            df['kAB'] = [self.parse.output['kAB']]
            df['kBA'] = [self.parse.output['kBA']]
            df['T'] = [value]
            dfs.append(df)
            print(f"Computed rate constants for temperature {value}") 
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        bigdf['numInA'] = self.parse.numInA
        bigdf['numInB'] = self.parse.numInB
        #if file exists, append to existing data
        if csv.is_file():
            olddf = pd.read_csv(csv)
            bigdf = olddf.append(bigdf)
        #write updated file to csv
        bigdf.to_csv(csv, index=False)

    def remove_output(self):
        """Delete PATHSAMPLE log files."""
        for f in glob.glob(self.path/'out*'):
            if Path(f).exists():	
                Path(f).unlink() 

""" Functions for using the ScanPathsample and ParseSample classes to perform
useful tasks. """

def scan_product_states(numinAs, numinBs, temps):
    """Calculate kNSS using NGT for various definitions of A and
    B sets."""
    if len(numinAs) is not len(numinBs):
        raise ValueError('numinAs and numinBs must have the same shape')
    
    for i in range(len(numinAs)):
        suffix = 'kNGT_ABscan'
        scan = ScanPathsample('/scratch/dk588/databases/LJ38.2010/10000.minima/pathdata', suffix=suffix)
        #since min.A/min.B have been sorted, simply change scan.parse.numInA/B
        scan.parse.define_A_and_B(numinAs[i], numinBs[i], sorted=True, mindata=scan.path/'min.data')
        #write new min.A/B files with first line changed
        scan.parse.write_minA_minB(scan.path/'min.A',scan.path/'min.B')
        scan.scan_temp('TEMPERATURE', temps)
        scan.remove_output()


if __name__=='__main__':
    temps = np.arange(0.03, 0.16, 0.01)
    #temps = np.array([0.03, 0.04])
    numinAs = [1, 1, 1, 1, 1,  1,   1,   1,   1,   1]
    numinBs = [1, 2, 3, 5, 7, 10, 100, 200, 300, 395]
    parse = ParsedPathsample('/scratch/dk588/databases/LJ38.2010/10000.minima/pathdata')
    #parse.sort_A_and_B(parse.path/'min.A.master',parse.path/'min.B.master', parse.path/'min.data')
    os.system(f"{PATHSAMPLE} > test")
    scan_product_states(numinAs, numinBs, temps)
    """
    for temp in temps:
        suffix = 'ABBA_1inA_1inB'
        scan = ScanPathsample('./pathdata', temp, suffix=suffix, outbase='out.NGT')
        scan.parse.append_input('TEMPERATURE', temp)
        nrgthreshs = np.linspace(0.01, 2.0, 100)
        scan.scan_param('REGROUPFREE', nrgthreshs.tolist(), outputkey='kSSBA')
    """
