module Bayes
include("./helpermodules.jl") 
using Pkg, CSV, DataFrames, MLDataUtils, Languages, .Sentiment140DataFrame, .TweetFormat

# Functions visible to external modules
export NaiveBayes, predict_positive, train, test, savenb, loadnb

### NAIVE BAYES PROBABILITY ALGORITHM ###
# Struct for naive bayes model
mutable struct NaiveBayes
    pos_total::Int
    neg_total::Int
    pos_occurences::Dict
    neg_occurences::Dict
end

# Functions
# returns true if the given tweet is predicted to be positive
function predict_positive(tweet, model)
    # Variables
    pos_total = model.pos_total
    neg_total = model.neg_total
    pos_occurences = model.pos_occurences
    neg_occurences = model.neg_occurences

    # Helper functions
    # number of times word appears with negative sentiment
    neg_count(word) = haskey(neg_occurences, word) ? neg_occurences[word] : 0
    # number of times word appears with positive sentiment
    pos_count(word) = haskey(pos_occurences, word) ? pos_occurences[word] : 0
    # probability of seeing word given negative sentiment, add 1 to avoid zero frequency problem
    prob_word_neg(word) = (neg_count(word) + 1)/neg_total
    # probability of seeing word given positive sentiment, add 1 to avoid zero frequency problem
    prob_word_pos(word) = (pos_count(word) + 1)/pos_total
    # probability sentiment is negative given a tweet
    prob_neg(tweet) = prod(prob_word_neg.(split(tweet))) * neg_total
    # probability sentiment is positive given a tweet
    prob_pos(tweet) = prod(prob_word_pos.(split(tweet))) * pos_total

    # compare both probabilities and return true or false
    prob_pos(tweet) > prob_neg(tweet)
end


### TRAINING ###
# Variables
const positive = 1

# Functions
is_pos_tweet(row) = parse(Int, string(row.class_label)) == positive
function train(set)
    # Helper functions
    add_neg_word(word) = !haskey(neg_occurences, word) ? neg_occurences[word] = 1 : neg_occurences[word] += 1
    add_pos_word(word) = !haskey(pos_occurences, word) ? pos_occurences[word] = 1 : pos_occurences[word] += 1
    neg_word(word) = (add_neg_word(word); neg_total += 1)
    pos_word(word) = (add_pos_word(word); pos_total += 1)
    train_word(word, pos) = pos ? pos_word(word) : neg_word(word)
    train_row(row) = train_word.(split(removenoise(string(row.tweet))), is_pos_tweet(row))

    # Variables 
    pos_total = 0
    neg_total = 0
    pos_occurences = Dict{AbstractString,Int}()
    neg_occurences = Dict{AbstractString,Int}()

    # train each row in set
    for row in eachrow(set)
        train_row(row)
    end

    # create NaiveBayes model
    return NaiveBayes(pos_total, neg_total, pos_occurences, neg_occurences)
end


### TESTING ###
# Functions
function test(set, model)
    correct(row) = predict_positive(removenoise(string(row.tweet)), model) == is_pos_tweet(row)

    num_correct = 0
    for row in eachrow(set)
        num_correct += correct(row) ? 1 : 0
    end
    num_correct/size(set)[1]
end

### SAVING ###
# Functions
function savenb(model, filename)
    open(filename, "w") do io
        write(io, string(model.pos_total) * "\n")
        write(io, string(model.neg_total) * "\n")
        write(io, string(length(model.pos_occurences)) * "\n")
        for pair in model.pos_occurences
            write(io, string(pair.first) * " " * string(pair.second) * "\n")
        end
        write(io, string(length(model.neg_occurences)) * "\n")
        for pair in model.neg_occurences
            write(io, string(pair.first) * " " * string(pair.second) * "\n")
        end
    end
end

### LOADING ###
# Functions
function loadnb(filename)
    pos_total = 0
    neg_total = 0
    pos_occurences = Dict{AbstractString,Int}()
    neg_occurences = Dict{AbstractString,Int}()

    open(filename, "r") do io
        pos_total = parse(Int, readline(io))
        neg_total = parse(Int, readline(io))
        # number of pairs in pos_occurences dictionary
        pos_pairs = parse(Int, readline(io))
        for i = 1:pos_pairs
            key, val = split(readline(io))
            pos_occurences[key] = parse(Int, val)
        end
        neg_pairs = parse(Int, readline(io))
        for i = 1:neg_pairs
            key, val = split(readline(io))
            neg_occurences[key] = parse(Int, val)
        end
    end

    return NaiveBayes(pos_total, neg_total, pos_occurences, neg_occurences)
end
end