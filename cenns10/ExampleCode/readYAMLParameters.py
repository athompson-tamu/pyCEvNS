#
# readYAMLParameters.py
#
#
# A simple demonstration of reading the YAML parameter file associated 
# with the data release from the COHERENT Collaboration pertaining to arXiv:2003.10630, taken from the initial CsI data release from the collaboration  
#
# Command line arguments specify the data file to read
# default values allow one to simply run 'python readYAMLParameters.py' from within the 'code' directory of the release
# this will read data stored in the 'data' directory of the release
# 
# created 2018 March, Grayson C. Rich
# gcrich@uchicago.edu
# 
# adapted for CENNS-10 data release June 2020, J. Zettlemoyer (jczettle@indiana.edu)

from __future__ import print_function
import yaml
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument('-inputFilename', type=str, default='../Data/LArParametersAnlA.yaml',
                        help='Name (with path, if in other directory) of the YAML file containing COHERENT experiment parameters')
parsedArgs = argParser.parse_args()
inputFilename = parsedArgs.inputFilename


parameterDictionary = yaml.load(open(inputFilename,'r'))

print('Printing names of the parameters included in YAML file..\n')
for entry in parameterDictionary:
    paramEntry = parameterDictionary.get(entry)
    paramName = paramEntry.get('name')
    paramValue = paramEntry.get('value')
    subparameters = paramEntry.get('parameters')
    
    if paramValue != None:
        print('{} has value {}'.format(paramName, paramValue))
    
    if subparameters:
        print('{} has list of parameters!'.format(paramName))
        for subparam in subparameters:
            subparamName = subparam.get('name')
            subparamValue = subparam.get('value')
            print('\t{} has value {}'.format(subparamName, subparamValue))
    
    print('\n')
