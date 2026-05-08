#!/bin/bash


rootDir=media/work/evan/GeneticInverseDesign
inputDir=home/evan/projects/NanostructureInverseDesignGA/testing/input



python3 nanostructureDesign.py -outDir output -targetFile /$inputDir/objScatteredPower0.txt -w0 325 -w1 650

python3 nanostructureDesign.py -outDir /$rootDir/outputProfile0 -targetFile /$inputDir/objScatteredPower0.txt -w0 325 -w1 650
python3 nanostructureDesign.py -outDir /$rootDir/outputProfile1 -targetFile /$inputDir/objScatteredPower1.txt -w0 325 -w1 650
python3 nanostructureDesign.py -outDir /$rootDir/outputProfile2 -targetFile /$inputDir/objScatteredPower2.txt -w0 400 -w1 700
python3 nanostructureDesign.py -outDir /$rootDir/outputProfile4 -targetFile /$inputDir/objScatteredPower4.txt -w0 420 -w1 675

rm -rf output/checkpoint
rm -rf /$rootDir/outputProfile0/checkpoint
rm -rf /$rootDir/outputProfile1/checkpoint
rm -rf /$rootDir/outputProfile2/checkpoint
rm -rf /$rootDir/outputProfile4/checkpoint