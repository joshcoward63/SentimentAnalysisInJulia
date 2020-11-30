include("./bayes.jl")
include("./helpermodules.jl")
using Pkg, CSV, DataFrames, MLDataUtils, Languages, .Bayes, .Sentiment140DataFrame, .TweetFormat

### MAIN PROGRAM ###
# Variables
path = "D:\\Downloads\\trainingandtestdata\\train.csv"

# Script
df = getdf(path)
training_set, testing_set = splitobs(shuffleobs(df), 0.9) 
model = train(training_set)
println(test(testing_set, model)) # output the accuracy rate to the console 
savenb(model, "naivebayes")