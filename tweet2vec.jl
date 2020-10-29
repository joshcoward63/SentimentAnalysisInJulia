
include("./helpermodules.jl")
using Pkg, CSV, DataFrames, Word2Vec, .Sentiment140DataFrame, .TweetFormat

### WORD2VEC CREATION ###
# Variables
path = "D:\\Downloads\\trainingandtestdata\\train.csv"

# Functions
createcorpus(tweets) = removestopwords(join(removenoise.(tweets)))
function writecorpusfile()
    tweets = getdf(path).tweet
    corpus = createcorpus(tweets)
    open("tweets", "w") do io
        write(io, corpus)
    end
end
createword2vec() = word2vec("tweets", "tweets-vec.txt", verbose = true)


### MAIN PROGRAM ###
writecorpusfile() # extract and format tweets from csv to create text corpus
createword2vec() # run text corpus through word2vec