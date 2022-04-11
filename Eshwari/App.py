from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/train')
def train():
    df = pd.read_excel('C:\Users\Admin\Downloads\Chemical_Industry\Chemical_Industry\Historical Alarm Cases xlsx')
    x = df.iloc[:,1:7]
    y = df['Spuriosity Index(0/1)']
    LR = LogisticRegression()
    LR.fit(x,y)
    joblib.dump(LR,'TrainedModel.pkl')
    return 'Model trained successfully'

@app.route('/test_model',methods = ['POST'])
def test():
    pkl_file = joblib.dump('TrainedModel')
    test_df = request.get_json
    t1 = test_df['Ambient Temperature']
    t2 = test_df['Calibration']
    t3 = test_df['Unwanted substance deposition']
    t4 = test_df['Humidity']
    t5 = test_df['H2S Content']
    t6 = test_df['detected by']
    my_test_df = [t1,t2,t3,t4,t5,t6]
    mydf_array = np.array(my_test_df )
    test_array = mydf_array.reshape(1,6)
    df_test = pd.DataFrame(test_array,
                           columns=['Ambient Temperature','Calibration','Unwanted substance deposition','Humidity','H2S Content','detected by]')
    y_pred = pkl_file.predict(df_test)
    if y_pred == 1:
        return "False Alarm,No Danger"
    else:
        return "True Alarm,Danger"

app.run(port=7000)
