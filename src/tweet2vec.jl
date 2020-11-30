
include("./helpermodules.jl")
using Pkg, CSV, DataFrames, Word2Vec, .Sentiment140DataFrame, .TweetFormat

### WORD2VEC CREATION ###
# Variables
path = "../data/train.csv"

# Functions
createcorpus(tweets) = join(removenoise.(tweets))
function writecorpusfile()
    tweets = getdf(path).tweet
    corpus = createcorpus(tweets)
    open("../data/tweets", "w") do io
        write(io, corpus)
    end
end
createword2vec() = word2vec("../data/tweets", "../data/tweets-vec.txt", verbose = true)


### MAIN PROGRAM ###
writecorpusfile() # extract and format tweets from csv to create text corpus
createword2vec() # run text corpus through word2vec