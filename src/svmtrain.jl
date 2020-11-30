include("./helpermodules.jl") 
using Pkg, CSV, DataFrames, Word2Vec, Languages, MLDataUtils, WordTokenizers, 
    Random, LIBSVM, Statistics, JLD, .Sentiment140DataFrame, .TweetFormat

### SVM INPUT INITILIZATION ###
# Functions
# removes noise from tweets and updates the tweet column of dataframe
denoise_df!(df) = df.tweet .= (removestopwords ∘ removenoise).(df.tweet)

# creates feature vector for tweet by averaging each word vector
function createFeatureVector(sentence)
    wordCount = 0
    vector = zeros(size(model)[1])
    sentence_split = split(sentence)
    for word in sentence_split
        try
            vector =  vector + get_vector(model, word)
            wordCount = wordCount + 1
        catch
            nothing
        end
    end 
    if wordCount > 0
        vector = vector / wordCount
    end
    return vector
end

### MAIN PROGRAM ###
# Variables
csvpath = "../data/train.csv"
word2vecpath = "../data/tweets-vec.txt"

# Script
# initial setup
df = getdf(csvpath)
model = wordvectors(word2vecpath)
denoise_df!(df)
df = DataFrame(shuffle(eachrow(df)))

# create feature vectors for tweets
features = hcat(createFeatureVector.(df.tweet)...) # hcat converts Array{Array{Float64,1},1} into Array{Float64,2}
labels = df.class_label

# divide model into training and testing sets
label_train = labels[1:10000]
label_test = labels[10001:20000]
feature_train = features[:,1:10000]
feature_test = features[:,10001:20000]

# train model
model = svmtrain(feature_train, label_train)

# test Model
(predicted_labels, decision_values) = svmpredict(model,feature_test)

# print model accuracy to console
print("Accuracy: ")
println(mean((predicted_labels .== label_test))*100)