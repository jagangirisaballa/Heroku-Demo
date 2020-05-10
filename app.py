from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import nltk
nltk.download('stopwords')
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

import string

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("https://raw.githubusercontent.com/jagangirisaballa/Heroku-Demo/master/data/YoutubeSpamMergeddata.csv", encoding='latin-1')
	df_data = df[["CONTENT","CLASS"]]

	#define a function to get rid of stopwords present in the messages
	def message_text_process(mess):
		no_punctuation = [char for char in mess if char not in string.punctuation]
		
		no_punctuation = ''.join(no_punctuation)
		
		return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]

	#use bag of words by applying the function and fit the data into it
	bag_of_words_transformer = CountVectorizer(analyzer=message_text_process).fit(df_data['CONTENT'])

	content_bagofwords = bag_of_words_transformer.transform(df_data['CONTENT'])

	#apply tfidf transformer and fit the bag of words into it (transformed version)
	tfidf_transformer = TfidfTransformer().fit(content_bagofwords)

	#print shape of the tfidf 
	content_tfidf = tfidf_transformer.transform(content_bagofwords)

	#choose naive Bayes model to detect the spam and fit the tfidf data into it
	from sklearn.naive_bayes import MultinomialNB
	spam_detect_model = MultinomialNB().fit(content_tfidf, df_data['CLASS'])


	#check model for the predicted and expected value say for message#2 and message#5
	content = df_data['CONTENT']
	cv = CountVectorizer()
	clf = MultinomialNB()
	bag_of_words_for_content = bag_of_words_transformer.transform([content])
	tfidf = tfidf_transformer.transform(bag_of_words_for_content)

	# # Features and Labels
	# df_x = df_data['CONTENT']
	# df_y = df_data.CLASS
    # # Extract Feature With CountVectorizer
	# corpus = df_x
	# cv = CountVectorizer()
	# X = cv.fit_transform(corpus) # Fit the Data
	# from sklearn.model_selection import train_test_split
	# X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.2, random_state=42)
	# #Naive Bayes Classifier
	# from sklearn.naive_bayes import MultinomialNB
	# clf = MultinomialNB()
	# clf.fit(X_train,y_train)
	
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = spam_detect_model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)










# from flask import Flask,render_template,url_for,request
# import pandas as pd 
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib





# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
# 	#url = 'https://drive.google.com/open?id=1Yr3Vjuzw9_s-wDsGJ5O0lehFlYfTuQKe_Sgd0WX4IxE'
# 	#df = pd.read_csv(StringIO(url), error_bad_lines=False)
# 	#df = pd.read_html(url,index_col=0)
# 	df= pd.read_csv("https://raw.githubusercontent.com/jagangirisaballa/Heroku-Demo/master/data/YoutubeSpamMergeddata.csv", encoding='latin-1')
# 	df_data = df[["CONTENT","CLASS"]]
# 	# Features and Labels
# 	df_x = df_data['CONTENT']
# 	df_y = df_data.CLASS
#     # Extract Feature With CountVectorizer
# 	corpus = df_x
# 	cv = CountVectorizer()
# 	X = cv.fit_transform(corpus) # Fit the Data
# 	from sklearn.model_selection import train_test_split
# 	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.2, random_state=42)
# 	#Naive Bayes Classifier
# 	from sklearn.naive_bayes import MultinomialNB
# 	clf = MultinomialNB()
# 	clf.fit(X_train,y_train)
	
# 	if request.method == 'POST':
# 		comment = request.form['comment']
# 		data = [comment]
# 		vect = cv.transform(data).toarray()
# 		my_prediction = clf.predict(vect)
# 	return render_template('result.html',prediction = my_prediction)



# if __name__ == '__main__':
# 	app.run(debug=True)
