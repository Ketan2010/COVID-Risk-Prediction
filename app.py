# importing Flask and other modules 
from flask import Flask, request, render_template 
import numpy as np
import pandas as pd


l1=['O-blood-group','child-adult','medical-complication','international-travel','contact-with-covid','covid-symptoms']

disease=['Very-high','Pretty-High','High','Pretty-low','low']

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("dataset/test.csv")

df.replace({'prognosis':{'Very-high':0,'Pretty-High':1,'High':2,'Pretty-low':3,'low':4}},inplace=True)

# print(df.head())

X= df[l1]


y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("dataset/train.csv")
tr.replace({'prognosis':{'Very-high':0,'Pretty-High':1,'High':2,'Pretty-low':3,'low':4}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

# Flask constructor 
app = Flask(__name__) 

def DecisionTree(psymptoms):
    
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # print("Decision Tree Accuracy: ", accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    inputtest = [psymptoms]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):# assigning value for variable "a" while checking the output number exists line 20 
        if(predicted == a):
            h='yes' 
            break
        
    if (h=='yes'):
        # print("COVID-19 Risk: ", disease[a])
        return(disease[a], accuracy_score(y_test, y_pred,normalize=False)) 
    else:
        return("Not Found", "Not Found")


def NaiveBayes(psymptoms):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # print("Naive Bayes Accuracy: ", accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------


    inputtest = [psymptoms]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        # print("COVID-19 Risk: ", disease[a])
        return(disease[a], accuracy_score(y_test, y_pred,normalize=False)) 
    else:
        return("Not Found", "Not Found")


@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        name = request.form.get("name")
        age = int(request.form.get("age"))
        blood_group = int(request.form.get("blood_group"))
        medical_complication = int(request.form.get("medical_complication"))
        travel_history = int(request.form.get("travel_history"))
        covid_contact = int(request.form.get("covid_contact"))
        covid_symptoms = int(request.form.get("covid_symptoms"))
        psymptoms = [age, blood_group, medical_complication, travel_history, covid_contact, covid_symptoms]
        naive_prediction , nb_accuracy = NaiveBayes(psymptoms)
        d_t_prediction, dt_accuracy = DecisionTree(psymptoms)
        
    return render_template("index.html", name = name, prediction_nb = naive_prediction,accuracy_nb = nb_accuracy , prediction_dt = d_t_prediction, accuracy_dt = dt_accuracy)
	
 
if __name__=='__main__':  
    app.run(debug=True) 
