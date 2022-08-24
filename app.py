
from pyexpat import model
from flask import Flask,render_template,request,url_for,redirect
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
app=Flask(__name__)
#load model
pred_model=pickle.load(open('model.pkl','rb'))
loaded_vect=pickle.load(open('vectoriser.pickle','rb'))
#extract features from text
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')


@app.route('/')
def hello():
    return render_template('index.html')

    
@app.route('/check',methods=['POST','GET'])
def predict():
    if request.method=="POST":
        email_check=request.form.get("email")
    print(email_check)
    ans=pred_model.predict(loaded_vect.transform([email_check]))
    print(ans[0])
    #return render_template('index.html')
    if ans[0]>=0.5:
        return render_template('index.html',pred="Mail is not spam")
    else:
        return render_template('index.html',pred="Mail is spam")

if __name__=='__main__':
    app.run(debug=True)
