#!/bin/bash

ffmpeg -f image2 -framerate 15 -i rmseTest/gen%dTopPerformers.png -loop -1 TopPerformerEvolutionRMSE.mp4
ffmpeg -f image2 -framerate 15 -i mseTest/gen%dTopPerformers.png -loop -1 TopPerformerEvolutionMSE.mp4
ffmpeg -f image2 -framerate 15 -i maeTest/gen%dTopPerformers.png -loop -1 TopPerformerEvolutionMAE.mp4
ffmpeg -f image2 -framerate 15 -i mreTest/gen%dTopPerformers.png -loop -1 TopPerformerEvolutionMRE.mp4
ffmpeg -f image2 -framerate 15 -i gapTest/gen%dTopPerformers.png -loop -1 TopPerformerEvolutionGap.mp4

ffmpeg -f image2 -framerate 15 -i rmseTest/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolutionRMSE.mp4
ffmpeg -f image2 -framerate 15 -i mseTest/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolutionMSE.mp4
ffmpeg -f image2 -framerate 15 -i maeTest/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolutionMAE.mp4
ffmpeg -f image2 -framerate 15 -i mreTest/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolutionMRE.mp4
ffmpeg -f image2 -framerate 15 -i gapTest/gen%dBottomPerformers.png -loop -1 BottomPerformerEvolutionGap.mp4

ffmpeg -f image2 -framerate 15 -i rmseTest/gen%dfitnessGen%d.png -loop -1 FitnessEvolutionRMSE.mp4
ffmpeg -f image2 -framerate 15 -i mseTest/gen%dfitnessGen%d.png -loop -1 FitnessEvolutionMSE.mp4
ffmpeg -f image2 -framerate 15 -i maeTest/gen%dfitnessGen%d.png -loop -1 FitnessEvolutionMAE.mp4
ffmpeg -f image2 -framerate 15 -i mreTest/gen%dfitnessGen%d.png -loop -1 FitnessEvolutionMRE.mp4
ffmpeg -f image2 -framerate 15 -i gapTest/gen%dfitnessGen%d.png -loop -1 FitnessEvolutionGap.mp4

rm -rf rmseTest/
rm -rf mseTest/
rm -rf maeTest/
rm -rf mreTest/
rm -rf gapTest/