include("./bayes.jl")
include("./helpermodules.jl")
using Gtk, Gtk.ShortNames ,Word2Vec, LIBSVM, JLD, SVR, .Bayes, 
    .Sentiment140DataFrame, .TweetFormat, DataFrames, CSV

### SETUP ###
# Variables
csvpath = "../data/train.csv"
nbpath = "../data/naivebayes"
word2vecpath = "../data/tweets-vec.txt"

# Functions
# calculates the n most similar vectors to a tweet
function cosine(vectors, tweet, n=10)
    metrics = vectors'*tweet
    topn_positions = sortperm(metrics[:], rev = true)[1:n]
    topn_metrics = metrics[topn_positions]
    return topn_positions, topn_metrics
end

# returns list of n most similar tweets
function cosine_similar_tweets(vectors, tweets, tweet, n=10)
    index, metr = cosine(vectors, tweet, n)
    return tweets[index]
end

# calculates magnitude of vector
mag(x) = sqrt(sum(x.^2))

# creates feature vector for a tweet with magnitude of 1
function createFeatureVector(sentence)
    wordCount = 0
    vector = zeros(size(word2vec_model)[1])
    sentence = split(sentence)
    for word in sentence
        try
            vector =  vector + get_vector(word2vec_model, word)
            wordCount = wordCount + 1
        catch
            nothing
        end
    end
    if wordCount > 0
        vector = vector / mag(vector)
    end
    return vector
end

# Script
model = loadnb(nbpath)
word2vec_model = wordvectors(word2vecpath)
df = getdf(csvpath)
denoised_tweets = removenoise.(df.tweet)
features = hcat(createFeatureVector.(denoised_tweets)...) # hcat converts Array{Array{Float64,1},1}


### GUI COMPONENTS ###
# Main Window
mainWin = GtkWindow("Home", 400, 160,visible = false)
vboxMain = GtkBox(:v)
hboxMain1 = GtkBox(:h)
sentimentID = GtkButton("\n\n    Sentiment Identifier      \n\n")
screen = Gtk.GAccessor.style_context(sentimentID)
similarPhrase = GtkButton("\n\n     Find a similar phrase    \n\n")
mainMenuLabel = GtkLabel("Main Menu")
push!(hboxMain1,sentimentID)
push!(hboxMain1,similarPhrase)
push!(vboxMain, mainMenuLabel)
push!(vboxMain, hboxMain1)
push!(mainWin, vboxMain)

# Sentiment Analysis Window
sentWin = GtkWindow("Sentiment Identifier", 475,150,visible = false)
prompt = GtkLabel("")
GAccessor.markup(prompt,"""<b>Please enter in a sentence or phrase that you wish to get the sentiment for below.</b>\n""")
#User entry
ent = GtkEntry()
set_gtk_property!(ent,:text, "Enter text here... ")

# set_gtk_property!(ent,:default-width, 400)
enterButton = GtkButton("Enter")
hbox = GtkBox(:h)
push!(hbox, ent)
push!(hbox, enterButton)

label = GtkLabel("Your text contains a ... sentiment.")
vbox = GtkBox(:v)
homeButton = GtkButton("Home")
push!(vbox,prompt)
push!(vbox, hbox)
push!(vbox, label)
push!(vbox, homeButton)
push!(sentWin, vbox)
function on_enter_click(w)
    str = get_gtk_property(ent,:text,String)
    if predict_positive(str, model) == true
        GAccessor.text(label,"Your text contains a positive sentiment.")
    else
        GAccessor.text(label,"Your text contains a negative sentiment.")
    end
end
signal_connect(on_enter_click, enterButton,"clicked")

# Similar Phrase Window
simPhraseWin = GtkWindow("Similar Phrase", 400, 220,visible = false)
prompt2 = GtkLabel("")
GAccessor.markup(prompt,"""<b>Please enter in a sentence or phrase that you wish to find similar examples of.</b>\n""")
#User entry
ent2 = GtkEntry()
set_gtk_property!(ent2,:text, "Enter text here... ")

# set_gtk_property!(ent,:default-width, 400)
enterButton2 = GtkButton("Enter")
hbox2 = GtkBox(:h)
push!(hbox2, ent2)
push!(hbox2, enterButton2)

label2 = GtkLabel("")
vbox2 = GtkBox(:v)
homeButton2 = GtkButton("Home")
push!(vbox2,prompt2)
push!(vbox2, hbox2)
push!(vbox2, label2)
push!(vbox2, homeButton2)
push!(simPhraseWin, vbox2)
function on_enter_click2(w)
    str = get_gtk_property(ent2,:text,String)
    tweet_vector = createFeatureVector(removenoise(str))
    similar_tweets = cosine_similar_tweets(features, df.tweet, tweet_vector, 5)
    similar_tweet_label = join(similar_tweets, "\n")
    GAccessor.text(label2,string(similar_tweet_label))
end
signal_connect(on_enter_click2, enterButton2,"clicked")

# Functions for Changing Windows
function on_sentID_click(w)
    visible(mainWin, false)
    Gtk.showall(sentWin)
end
function on_simPhrase_click(w)
    visible(mainWin, false)
    Gtk.showall(simPhraseWin)
end
function on_home_click(w)
    visible(sentWin, false)
    Gtk.showall(mainWin)
end   
function on_home_click2(w)
    visible(simPhraseWin, false)
    Gtk.showall(mainWin)
end   
signal_connect(on_home_click2, homeButton2, "clicked")
signal_connect(on_home_click, homeButton, "clicked")
signal_connect(on_sentID_click, sentimentID,"clicked")
signal_connect(on_simPhrase_click, similarPhrase,"clicked")


### MAIN PROGRAM ###
Gtk.showall(mainWin)  