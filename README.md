Disaster Tweets Neuro-Linguistic Programming: Exploratory Data Analysis, Application of BERT Using Transformers Library With Pytorch

This notebook includes: Preprocessing the text, visualizing the processed data by several methods like tweet lenghts, 
word counts, average word lengths, ngrams etc. I especially wanted to clean data before I visualize it, 
perhaps you should investigate the raw data you got first then move to cleaning in normal cases 
but I didn't want to pile it on so I went this way. Then I used some more analysis tecniques like Word Clouds, NER's etc. 
to give us different angles to look from. At the last part we're going to implement BERT model to do tokenization, classification 
and prediction with using transformers. 

Loading the Data :

 I added "v"'s at the end of our variables for visualization because some of the pre-processing are not needed for the modelling but we can use them for our EDA part.
 
 Cleaning Text : 
 
 Before we visualize our text data I wanted to make it look better with some general helper functions to clear out things like:
 urls, emojis, html tags, punctuations... We'll add all of them in one column called 'text_clean' then move from there for next steps.
 When we have cleaner text we can apply our tokenizer to split each word into a token. I'll apply this and next steps to individual columns 
 to show each step of our progress. Next we transforming all words to lowercase then we remove stopwords (they don't mean much in sentence alone) 
 so we use NLTK stopwords for it.After removing these words we gonna lemmatize them but for that we need to add some extra steps to 
 do it properly: We gonna apply part of speech tags to our text (like verb, noun etc.) then we convert them to wordnet format and 
 finally we can apply lemmatizer and save it to 'lemmatized' column. And one last thing we convert these tokenized lists back to str version for future uses.

Visualizing the Data : 

When we check our target variables and look at how they disturbuted we can say it not bad. There is no huge difference between classes we can say it's good sign for modelling.

Let's start with the number of characters per tweet and compare if it's disaster related or not. 
It seems disaster tweets are longer than non disaster tweets in general. We can assume longer tweets are more 
likely for disasters but this is only an assumption and might be not true...

let's check number of words per tweet now, they both look somewhat normally distributed, again disaster 
tweets seems to have slightly more words than non disaster ones. We might dig this deeper to get some more info in next part..

This time we're gonna check if word complexity differs from tweet class. 
It looks like disaster tweets has longer words than non disaster ones in general. 
It's pretty visible which is good sign, yet again we can only assume at this stage..


We start with most common words in both classes. 
I'd say it's pretty obvious if it's from disaster tweets or not. 
Disaster tweets has words like fire, kill, bomb indicating disasters. 
Meanwhile non disaster ones looks like pretty generic.

Again it's pretty obvious to seperate two classes if it's disaster related or not. 
There are some confusing bigrams in non disaster ones like body bag, emergency service etc. 
which needs deeper research but we'll leave it here since we got what we looking for in general.

Things are much clearer with sequences of 3 words. 
The confusing body bags were cross body bags (Who uses them in these days anyways!) which I found it 
pretty funny when I found the reason of the confusion. Anyways we can see disasters are highly 
seperable now from non disaster ones, which is great!

We'll be using a method called Non-Negative Matrix Factorization (NMF) to see if we can get some defined topics out of our 
TF-IDF matrix, with this way TF-IDF will decrease impact of the high frequency words, so we might get more specific topics.

When we inspect our top ten topics we might need to use little imagination to help us understand them. 
Well actually they are pretty seperable again, I'd say disaster topics are much more clearer to read, 
we can see the topics directly by looking at them, meanwhile non disaster ones are more personal topics...

Wordclouds are popular approach in NLP tasks. We're going to use the library exactly designed for it called "WordCloud", 
I also wanted to mask it with twitter logo shape and grey colors just for adding more interesting presentation and 
show what you can do with this library. When we look our word clouds we can clearly say which 
one is disaster on which one is not. Pretty good!

One last thing before we move on the modelling is Named Entity Recognition. It's a method for extracting 
information from text and returns which entities that are present in the text are classified into 
predefined entity types like "Person", "Place", "Organization", etc. By using NER we can get great 
insights about the types of entities present in the given text dataset.


When we look our NER results we can get lots of great insights. We can see that in disaster tweets countries, 
cities, states are much more common than non disaster ones. Again nationality or religious or political group 
names are more likely to be mentioned in disaster tweets. These are great indicators for us...

Building the Bert Model : 

Tokenization and Formatting the Inputs
For feeding our text to BERT we have to tokenize our text first and then these tokens must be mapped. For this job we gonna download and use BERT's own tokenizer. Thanks to Transformers library it's like one line of code, we also convert our tokens to lowercase for uncased model. You can see how the tokenizer works below there on first row of tweets for example.
We set our max len according to our tokenized sentences for padding and truncation, then we use tokenizer.encode_plus it'll split the sentences into tokens, then adds special tokens for classificication [CLS]:
The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. (from the BERT paper)

Then it adds [SEP] tokens for making BERT decide if sentences are related. In our case it shouldn't be that important I think.
Then our tokenizer map's our tokens to their IDs first and pads or truncates all sentences to same length according to our max length. If sentence is longer than our limit it gets truncated, if it's shorter than our defined length then it adds [PAD] tokens to get them in same length.
Finally tokenizer create attention masks which is consisting of 1's and 0's for differentiating [PAD] tokens from the actual tokens.
We do these steps for each train and test set and then get our converted data for our BERT model. We also split train test on our train data for checking our models accuracy.
Lastly we define how to load the data into our model for training, since we can't use it all at once because of memory restrictions. On the official BERT paper batch size of 16 or 32 is recommended.


