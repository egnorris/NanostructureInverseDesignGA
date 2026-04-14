#!/bin/bash

for ft in {"rmse",'mse','mae','mre','gap'}; do

    directory="$ft-TestGauss"
    python3 run.py -out $directory -tp 'objScatteredPower2.txt' -ng 150 -ft $ft -seed 0

    ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 TopPerformerEvolution-$ft-gauss.mp4
    ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-gauss.mp4
    rm -rf $directory/gen*

    directory="$ft-TestMultiPeak"
    python3 run.py -out $directory -tp 'objScatteredPower1.txt' -ng 150 -ft $ft -seed 0

    ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dTopPerformers.png -loop -1 TopPerformerEvolution-$ft-multipeak.mp4
    ffmpeg -y -f image2 -framerate 15 -i $directory/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolution-$ft-multipeak.mp4
    rm -rf $directory/gen*

done

