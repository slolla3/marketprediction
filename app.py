# importing the packages
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, flash#, redirect, url_for, session, logging
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
# value of __name__ should be  '__main__'
app = Flask(__name__)
# Loading model so that it works on production 
model = joblib.load('./model/model.pkl')
global df
@app.route('/')
def index():
	# Index page
	return render_template('index.html', nfeat=model.n_features_)

@app.route('/load')
def load():
	global df
	df = pd.read_csv("./data/Social_Network_Ads.csv")
	#df.Age = df.Age.astype(int) 
	return render_template('showdata.html', msg='Original Data', data=df.to_html(classes='table'))

global df_train, df_test, y_train, y_test

@app.route('/preprocess')
def preprocess():
	global df
	global df_train, df_test, y_train, y_test
	# df = pd.read_csv("./data/Social_Network_Ads.csv")
	
	#defining predictors and label columns to be used
	predictors = ['Gender', 'Age', 'EstimatedSalary', 'Product']
	label = 'Purchased'

	#Splitting data into training and testing
	df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)

	# Data cleaning and filling missing values
	age_fillna = df_train.Age.mean()
	EstimatedSalary_fillna = df_train.EstimatedSalary.value_counts().index[0]

	# filling missing values in training data
	df_train.Age = df_train.Age.fillna(df.Age.mean())
	df_train.EstimatedSalary = df_train.EstimatedSalary.fillna(EstimatedSalary_fillna)

	# filling missing values imputed from training set to avoid data leakage
	df_test.Age = df_test.Age.fillna(df.Age.mean())
	df_test.EstimatedSalary = df_test.EstimatedSalary.fillna(EstimatedSalary_fillna)

	# Label encoding of object type predictors
	le = dict()
	for column in df_train.columns:
	    if df_train[column].dtype == np.object:
	        le[column] = LabelEncoder()
	df_train[column] = le[column].fit_transform(df_train[column])

	# Applying same encoding from training data to testing data
	for column in df_test.columns:
	    if df_test[column].dtype == np.object:
	df_test[column] = le[column].transform(df_test[column])

	df_train.Age = df_train.Age.astype(int)
	return render_template('showdata.html', msg='Shuffle and Preprocess Data', data=df_train.to_html(classes='table'))
@app.route('/train', methods=['GET'])
def train():
	global df_train, df_test, y_train, y_test
	# reading data
	# df = pd.read_csv("./data/Social_Network_Ads.csv")
	# Initializing the model
	model = RandomForestClassifier(n_estimators=25, random_state=42)

	# Fitting the model with training data
	model.fit(X=df_train, y=y_train)

	# Saving the trained model on disk
	joblib.dump(model, './model/model.pkl')

	# Return success message for user display on browser
	return render_template('trainsuccess.html', msg='Trained Successfully', data=df_train.to_html(classes='table'), model=model)
class PredictorsForm(Form):
	"""
	This is a form class to retrieve the input from user through form

	Inherits: request.form class
	"""
	Gender = StringField(u'Gender (0: Female and 1: Male)', validators=[validators.input_required()])
	Age = StringField(u'Age (For eg.: 24)', validators=[validators.input_required()])
	EstimatedSalary = StringField(u'EstimatedSalary (For eg.: 30000)', validators=[validators.input_required()])
	Product = StringField(u'Product (For eg.: IPHONE, IPOD...)', validators=[validators.input_required()])

@app.route('/accuracy')
def accuracy():
    global df
    global df_train, df_test, y_train, y_test
crossvalscorestrain = cross_val_score(model, df_train, y_train, scoring='accuracy') 
crossvalscorestest = cross_val_score(model, df_test, y_test, scoring='accuracy') 
    # Passing the predictions to new view(template)
    return render_template('accuracy.html', score1=crossvalscorestrain, score2=crossvalscorestest, a1=sum(crossvalscorestrain)/len(crossvalscorestrain),a2=sum(crossvalscorestest)/len(crossvalscorestest))



@app.route('/predict', methods=['GET', 'POST'])
def predict():
	form = PredictorsForm(request.form)
	
	# Checking if user submitted the form and the values are valid
	if request.method == 'POST' and form.validate():
		# Now save all values passed by user into variables
		Gender = form.Gender.data
		Age = form.Age.data
		EstimatedSalary = form.EstimatedSalary.data
		proid = form.Product.data
		proid = int(proid)        
		# proid=1
		if proid==1 : Product='IPHONE'
			
		if proid==2 : Product='VIVO'
			
		if proid==3 : Product='BOSE'
			
		if proid==4 : Product='MAC'
			
		if proid==5 : Product=='SURFACE'
			
		if proid==6 : Product=='FITBIT'
			
		if proid==7 : Product=='IPOD'
			
		if proid==8 : Product=='VAIO'
			
		if proid==9 : Product=='WALKMAN'

		if proid==10 : Product=='IPAD'
			


		# Creating input for model for predictions
		predict_request = [int(1 if Gender=='male' else 0), int(Age), int(EstimatedSalary), proid]
		predict_request = np.array(predict_request).reshape(1, -1)

		# Class predictions from the model
		prediction = model.predict(predict_request)
		prediction = str(prediction[0])

		# Survival Probability from the model
		predict_prob = model.predict_proba(predict_request)
		predict_prob = str(predict_prob[0][1])

		# Passing the predictions to new view(template)
		return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob, Product=Product)

	return render_template('predict.html', form=form)


if __name__ == '__main__':
	# Load the pre-trained model from the disk
	# model = joblib.load('./model/model.pkl')
	# Running the app in debug mode allows to change the code and
	# see the changes without the need to restart the server
	app.run(debug=True)
