import numpy as np
from flask import Flask, request, jsonify, render_template

import joblib
# from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load saved models
dt_model = joblib.load('models/nate_decision_tree.sav')
# dl_model = joblib.load('models/imblearn_pipeline.sav')
# dl_model.named_steps['kerasclassifier'].model = load_model('models/keras_model.h5')
knn_model = joblib.load('models/nate_knn.sav')
lr_model = joblib.load('models/nate_logistic_regression.sav')
rf_model = joblib.load('models/nate_random_forest.sav')
svm_model = joblib.load('models/SVM_model.sav')
xgb_model = joblib.load('models/XGBoost_model.sav')

# Dictionary of all loaded models
loaded_models = {
    'dt': dt_model,
    #'dl': dl_model,
    'knn': knn_model,
    'lr': lr_model,
    'rf': rf_model,
    'svm': svm_model,
    'xgb': xgb_model
}

# Function to decode predictions 
def decode(pred):
    if pred == 1: return 'Customer Exits'
    else: return 'Customer Stays'

@app.route('/')
def home():
    # Initial rendering
    result = [{'model':'Decision Tree', 'prediction':' '},
              #{'model':'Deep Learning', 'prediction':' '},
              {'model': 'K-nearest Neighbors', 'prediction': ' '},
              {'model': 'Logistic Regression', 'prediction': ' '},
              {'model': 'Random Forest', 'prediction': ' '},
              {'model': 'SVM', 'prediction': ' '},
              {'model': 'XGBoost', 'prediction': ' '}]
    
    # Create main dictionary
    maind = {}
    maind['customer'] = {}
    maind['predictions'] = result

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():

    # List values received from index
    values = [x for x in request.form.values()]

    # new_array - input to models
    new_array = np.array(values).reshape(1, -1)
    print(new_array)
    print(values)
    
    # Key names for customer dictionary custd
    cols = ['CreditScore',
            'Geography',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary']

    # Create customer dictionary
    custd = {}
    for k, v in  zip(cols, values):
        custd[k] = v

    # Convert 1 or 0 to Yes or No    
    yn_val = ['HasCrCard', 'IsActiveMember']
    for val in  yn_val:
        if custd[val] == '1': custd[val] = 'Yes'
        else: custd[val] = 'No'

    # Loop through 'loaded_models' dictionary and
    # save predictiond to the list
    predl = []
    for m in loaded_models.values():
        predl.append(decode(m.predict(new_array)[0]))

    result = [
            {'model':'Decision Tree', 'prediction':predl[0]},
            #{'model':'Deep Learning', 'prediction':predl[1]},
            {'model': 'K-nearest Neighbors', 'prediction': predl[1]},
            {'model': 'Logistic Regression', 'prediction': predl[2]},
            {'model': 'Random Forest', 'prediction': predl[3]},
            {'model': 'SVM', 'prediction': predl[4]},
            {'model': 'XGBoost', 'prediction': predl[5]}
            ]

    # Create main dictionary
    maind = {}
    maind['customer'] = custd
    maind['predictions'] = result

    return render_template('index.html', maind=maind)


if __name__ == "__main__":
    app.run(debug=True)