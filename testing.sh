#!/bin/bash

n=100


for x in {0,1,2,3,4,5}; do

    for s in '4'; do
        for ft in {'mae','mse'}; do
            directory="Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 15 -fL 0.79 -l0 420 -l1 675
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            rm -rf $directory/gen*
        done
        ffmpeg -y -i p$s-mae-seed$x-Evo.mp4 -i p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l p$s-seed$x-EvoComparison.mp4
    done
done
"""
for x in {0,1,2,3,4,5}; do

    for s in {'0','1'}; do
        for ft in {'mae','mse'}; do
            directory="Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 15 -fL 0.99  -l0 325 -l1 650
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            rm -rf $directory/gen*
        done
        ffmpeg -y -i p$s-mae-seed$x-Evo.mp4 -i p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l p$s-seed$x-EvoComparison.mp4
    done

    ffmpeg -y -i p0-seed$x-EvoComparison.mp4 -i p1-seed$x-EvoComparison.mp4 -filter_complex vstack test1-seed$x-EvoComparison.mp4


    for s in '2'; do
        for ft in {'mae','mse'}; do
            directory="Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 15 -fL 0.79 -l0 400 -l1 700
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            rm -rf $directory/gen*
        done
        ffmpeg -y -i p$s-mae-seed$x-Evo.mp4 -i p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l p$s-seed$x-EvoComparison.mp4
    done



    for s in '3'; do
        for ft in {'mae','mse'}; do
            directory="Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 15 -fL 0.79 -l0 350 -l1 550
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            rm -rf $directory/gen*
        done
        ffmpeg -y -i p$s-mae-seed$x-Evo.mp4 -i p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l p$s-seed$x-EvoComparison.mp4
    done


    ffmpeg -y -i p2-seed$x-EvoComparison.mp4 -i p3-seed$x-EvoComparison.mp4 -filter_complex vstack test2-seed$x-EvoComparison.mp4

    for s in '4'; do
        for ft in {'mae','mse'}; do
            directory="Spectrum$s-$ft-Seed$x"
            python3 run.py -out $directory -tp "objScatteredPower$s.txt" -ng $n -nT 20 -nC 20 -nR 20 -nP 20 -nF 20 -seed $x -ft $ft -gL 15 -fL 0.79 -l0 420 -l1 675
            ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 p$s-$ft-seed$x-Evo.mp4
            #ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-profile$s-seed$x.mp4
            rm -rf $directory/gen*
        done
        ffmpeg -y -i p$s-mae-seed$x-Evo.mp4 -i p$s-mse-seed$x-Evo.mp4 -filter_complex hstack -loop -l p$s-seed$x-EvoComparison.mp4
    done

done


