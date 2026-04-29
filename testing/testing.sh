#!/bin/bash

n=100

rootDir=media/work/evan/GeneticInverseDesign

#RNG Seeds for performance comparison
for x in {0,1,2,3,4,5}; do
    #Target spectra for each test
    for s in {'0','1'}; do
        for ft in {'mae','mse'}; do
            directory="/$rootDir/Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 25 -fL 0.995  -l0 325 -l1 650 --numSave 10
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 /$rootDir/p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            #rm -rf $directory/gen*
        done
        ffmpeg -y -i /$rootDir/p$s-mae-seed$x-Evo.mp4 -i /$rootDir/p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l /$rootDir/p$s-seed$x-EvoComparison.mp4
    done

    ffmpeg -y -i /$rootDir/p0-seed$x-EvoComparison.mp4 -i /$rootDir/p1-seed$x-EvoComparison.mp4 -filter_complex vstack test1-seed$x-EvoComparison.mp4

    #Target spectra for each test
    for s in '2'; do
        for ft in {'mae','mse'}; do
            directory="/$rootDir/Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 25 -fL 0.8 -l0 400 -l1 700 --numSave 10
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 /$rootDir/p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            #rm -rf $directory/gen*
        done
        ffmpeg -y -i /$rootDir/p$s-mae-seed$x-Evo.mp4 -i /$rootDir/p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l /$rootDir/p$s-seed$x-EvoComparison.mp4
    done


    #Target spectra for each test
    for s in '4'; do
        for ft in {'mae','mse'}; do
            directory="/$rootDir/Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 25 -fL 0.8 -l0 420 -l1 675 --numSave 10
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 /$rootDir/p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            #rm -rf $directory/gen*
        done
        ffmpeg -y -i /$rootDir/p$s-mae-seed$x-Evo.mp4 -i /$rootDir/p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l /$rootDir/p$s-seed$x-EvoComparison.mp4
    done


    ffmpeg -y -i /$rootDir/p2-seed$x-EvoComparison.mp4 -i /$rootDir/p3-seed$x-EvoComparison.mp4 -filter_complex vstack test2-seed$x-EvoComparison.mp4



done


