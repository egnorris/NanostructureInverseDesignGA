import argparse

parser = argparse.ArgumentParser(description="Run Genetic Algorithm Inverse Design")
#Required Parameters
parser.add_argument('-out', '--outputDir', type = str, required=True,
    help="Set directory to save output data to.")

parser.add_argument('-tp', '--targetPath', type=str, default=None,
    help="Enter the path of the target scattered power spectrum; supercedes --targetGeneration argument if valid")

#Parameters with a default value
parser.add_argument('-ng', '--numGenerations', type = int, required=False, default=25,
    help="Define the number of generations to run the algorithm for; \nDefault: 25")
parser.add_argument('-ns', '--numSave', type=int, required=False, default = 6,
    help="Define the number of top performers to be saved from each generation to a .MAT binary; \n Default: 6")


parser.add_argument('-mD', '--modelDirectory', type=str, required=False, default = "/media/work/evan/deep_learning_data/trained_models",
    help="")

parser.add_argument('-nV', type=int, required=False, default = 24,
    help="")

parser.add_argument('-nT', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nR', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nC', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nP', type=int, required=False, default = 30,
    help="")

parser.add_argument('-nF', type=int, required=False, default = 30,
    help="")

parser.add_argument('-r0', '--rMin', type=int, required=False, default = 10,
    help="")

parser.add_argument('-r1', '--rMax', type=int, required=False, default = 75,
    help="")

parser.add_argument('--smoothness', type=int, required=False, default = 5,
    help="")

parser.add_argument('-p','--precision', type=int, required=False, default = 12,
    help="")

parser.add_argument('-mR', '--mutationRate', type=float, required=False, default = 0.1,
    help="")

parser.add_argument('-cP', '--numCrossoverPoints', type=int, required=False, default = 1,
    help="")

parser.add_argument('-l' , required=False, default = [1,2,2],
    help="")

parser.add_argument('-m' , required=False, default = [1,1,2],
    help="")

parser.add_argument('-f' , required=False, default = ['E','H'],
    help="")

parser.add_argument('-gL', '--gapLim' , required=False, type=int, default = 15,
    help="")
parser.add_argument('-fL', '--fitLim' , required=False, type=float, default = 0.995,
    help="")

parser.add_argument('-l0', '--minLambda' , required=False, type=float, default = 300,
    help="")

parser.add_argument('-l1', '--maxLambda' , required=False, type=float, default = 800,
    help="")

parser.add_argument('-sseW' ,'--sseFitnesWeight', required=False, type=float, default = 1,
    help="")

parser.add_argument('-ssaW' ,'--ssaFitnesWeight', required=False, type=float, default = 1,
    help="")

#Optional Parameters
parser.add_argument('-seed', '--seed', type = int, required=False, default=None,
    help="Define a seed for the random number generator if desired.")


kwargs = parser.parse_args().__dict__



print(kwargs)

