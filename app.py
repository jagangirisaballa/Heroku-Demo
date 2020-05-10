from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib





app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#url = 'https://drive.google.com/open?id=1Yr3Vjuzw9_s-wDsGJ5O0lehFlYfTuQKe_Sgd0WX4IxE'
	#df = pd.read_csv(StringIO(url), error_bad_lines=False)
	#df = pd.read_html(url,index_col=0)
	df= pd.read_csv("https://raw.githubusercontent.com/jagangirisaballa/Heroku-Demo/master/data/YoutubeSpamMergeddata.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.2, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
