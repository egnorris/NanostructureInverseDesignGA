#!/bin/bash

ffmpeg -f image2 -framerate 5 -i data/gen%dTopPerformers.png -loop -1 data/TopPerformerEvolution.mp4
ffmpeg -f image2 -framerate 5 -i data/gen%dBottomPerformers.png -loop -1 data/BottomPerformerEvolution.mp4
ffmpeg -f image2 -framerate 5 -i data/fitnessGen%d.png -loop -1 data/FitnessEvolution.mp4