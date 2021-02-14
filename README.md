# COVID-Risk-Prediction
This project is use to predict COVID-19 Risk on the basis of available data of India (Available in dataset folder) using 2 different data classification algorithms i.e Naive Bayes algorithm and Decision tree algorithm.

## Demo
This project is deployed on heroku.
predictor-covid.herokuapp.com/

## How to use
```
$ git clone https://github.com/Ketan2010/COVID-Risk-Prediction.git
$ mv COVID-Risk-Prediction covid
$ cd covid
$ pip install -r requirements.txt
$ app.py
```

## Directory Structure
```
├── dataset 
│   └── train.csv
|   └── test.csv
├── static 
│   └── images for frontend
├── templates 
│   └── index.html
├── Procfile
├── README.md
├── app.py
└── requirements.txt
```

## Technology Stack
* python3 with libraries: Flask, numpy, scikit-learn, pandas
* for frontend: HTML, CSS
