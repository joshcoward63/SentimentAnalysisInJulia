# SentimentAnalysisInJulia

## Installing Julia
1. To install Julia first navigate to the following ```https://julialang.org/downloads/```
2. Select the Julia version that corresponds with your given machine
3. Install Julia

## Installing development enviornment
1. Jupter notebook is required to run the code
2. To install Jupter notebook first download and install anaconda from the following ```https://docs.anaconda.com/anaconda/install/```
3. Choose the version that corresponds with you machine
4. Install Ananconda
5. Open Ananconda Navigator
6. Click on Juypter notebook to install it

## Installing required Julia packages
1. Type in ```Julia``` from start and click on Julia
2. Press ```]``` to enter package manager
3. Enter the following to install the required packages
  ```add IJulia```
  ```add Gtk```
  ```add DataFrames```
  ```add CSV```
  ```add LIBSVM```
  ```add Word2Vec```
  ```add Languages```
  ```add MLDataUtils```
  ```add WordTokenizers```
  ```add JLD```
  ```add BSON```
  ```add Random```
  ```add SVR```
  ```add Statistics```

## Required files

### Included files
SentimentAnalysis.ipynb - Main file in which data set is explored, cleaned and both word2vec and the SVM models are generated
SentimentAnalysisGUI.ipynb - Graphical User Interface users can get sentiment of inputted phrase/tweet as well as a list iof similar tweets
bayes.jl - Naive Bayes ML Algorithm Module
helpermodules.jl - 
naivebayes - Saved Naive Bayes model information
nbtrain.jl - Created Naive Bayes model and saves it to file
README - this file

### Files that need to be downloaded
train.csv - Sentiment140 data set can be downloaded from the following ```http://help.sentiment140.com/for-students```

### Files that are generated
tweets.txt - text corpus created from cleaned tweets to be passed into word2vec
tweets-vec.txt - saved word2vec model
naivebayes - Saved Naive Bayes model information

## Side notes
Video covering this project can be found here ```https://www.youtube.com/watch?v=Y-IN-KpO-64```
