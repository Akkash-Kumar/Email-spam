import pandas as pd   #read dataset

messages = pd.read_csv('EmailCollection', sep='\t', names=['LABEL', 'MESSAGES'])   #reading dataset seperated by tab space and giving column names

import seaborn as sns   #draw chart

import matplotlib.pyplot as plt  #plot chart

sns.countplot(x='LABEL', data=messages)   #count no of spam and ham messages
plt.show()  #show chart
 
import nltk   #nlp

import re  #regular expression

nltk.download('punkt')  #tokenization
nltk.download('stopwords')  #stopwords like is,are,was....
nltk.download('wordnet')   #find synonym or antonym

from nltk.stem import PorterStemmer  #stemming
from nltk.corpus import stopwords  #stopwords
from nltk.stem import WordNetLemmatizer  #lemmatization


#initialise
stemmer = PorterStemmer()
lemmatizer=WordNetLemmatizer()


corpus = []

#data preprocessing
for i in range(0, len(messages)):
    #Removes non-alphabetic characters.
    review = re.sub('[^a-zA-Z]', ' ', messages['MESSAGES'][i])
    #Converts the text to lowercase.
    review = review.lower()
    #Tokenizes the text into words.
    review = review.split()
    #Applies stemming using PorterStemmer.
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    #Joins the processed words back into a string (review),(complete sentence)
    review = ' '.join(review)
    corpus.append(review)
print(corpus)  #end

# spliting ip and op
from sklearn.feature_extraction.text import CountVectorizer  #change into vector values
cv = CountVectorizer(max_features = 3500)
X = cv.fit_transform(corpus).toarray()  #ip - vector values in array (numerical values 0 or 1)

y = pd.get_dummies(messages['LABEL'])   #show either true or false under each labels
y = y.iloc[:, 1].values #taking spam column values

##import pickle
# Creating a pickle file for the CountVectorizer
##pickle.dump(cv, open('cv-transform.pkl', 'wb'))

#split data for training and testing
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #80% for training and 20% for testing


##print("X_train",X_train)
##print("X_test",X_test)


#MultinomialNB

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

mnb = MultinomialNB(alpha=0.8)

mnb.fit(X_train,y_train)   #training


y_pred_mnb=mnb.predict(X_test)  #evaluation

mnb_acc = accuracy_score(y_pred_mnb,y_test)  #accuracy calc
print("MNB Accuracy",mnb_acc)


#model prediction
message='Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'  #sample ip
data = [message]
vect = cv.transform(data).toarray() #convert to vector values using count vectoriser in array 
my_prediction = mnb.predict(vect)  #predict op
if my_prediction==0:
    print("It's a Ham Mail")
else:
    print("It's a Spam Mail")


##WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
##Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...

    
# Creating a pickle file for the Multinomial Naive Bayes model
##filename = 'model.pkl'
##pickle.dump(mnb, open(filename, 'wb'))
