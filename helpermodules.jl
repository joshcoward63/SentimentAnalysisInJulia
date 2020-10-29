### CSV DATAFRAME MODULE ###
module Sentiment140DataFrame
using Pkg, CSV, DataFrames

# Functions visible to external modules
export getdf

# Variables
colnames = ["class_label", "id", "date", "flag", "userid", "tweet"]

# Functions
renamecolumns!(df, colnames) = names!(df, Symbol.(colnames))
formatclasslabels!(df) = (replace!(df.class_label, "4" => "1"); replace!(df.class_label, "0" => "-1"))
function getdf(filepath) 
    df = DataFrame(CSV.read(filepath, normalizenames = true))
    head(df)
    renamecolumns!(df, colnames)
    formatclasslabels!(df)
    return df
end
end


### TWEET FORMATTING MODULE ###
module TweetFormat
using Languages

# Functions visible to external modules
export removenoise, removestopwords

# Variables
usernames = r"@\w+" # regex for usernames such as '@twitter_user123'
links = r"((https?:\/\/)|(www\.))\S+" # regex for links starting with 'http' or 'www.'
htmltags = r"&#?\w+;" # regex for html character entities such as '&lt;' or '&quot;'
repeatedletters = r"(.)\1{3,}" # regex for letters repeated three or more times, such as "hiii"
apostrophes = "'" # separate so contractions don't get spaced out e.g. 'can't' becomes 'cant' not 'can t'
special = [usernames, links, htmltags, apostrophes, repeatedletters]
stopword_list = replace.(stopwords(Languages.English), "'" => "")

# Functions
remove(str, pat) = replace(str, pat => "")
removefromlist(list, pat) = filter(x->x!=pat, list)
removespecial(str) = reduce(remove, special, init=str)
format(str) = lowercase(join(split(replace(str, r"[^A-Za-z ]" => " ")), " "))*" "
removenoise(tweet) = (format ∘ removespecial)(tweet)
removestopwords(str) = join(reduce(removefromlist, stopword_list, init=split(str)), " ")
end