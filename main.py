
from tokenize import Intnumber
from fastapi import Form, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json




app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class diabetes_model_input(BaseModel):
    
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction :  float
    Age : int
    



# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : diabetes_model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']


    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])
    if prediction[0] == 0:
        return 0
    
    else:
        return 1
    




class heart_model_input(BaseModel):
    
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach:int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
  





# loading the saved model
heart_model = pickle.load(open('heart_model.sav','rb'))




@app.post('/heart_prediction')
def heart_pred(input_parameters:  heart_model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    


    age= input_dictionary['age']
    sex= input_dictionary['sex']
    cp= input_dictionary['cp']
    trestbps= input_dictionary['trestbps']
    chol= input_dictionary['chol']
    fbs= input_dictionary['fbs']
    restecg= input_dictionary['restecg']
    thalach=input_dictionary['thalach']
    exang= input_dictionary['exang']
    oldpeak= input_dictionary['oldpeak']
    slope= input_dictionary['slope']
    ca= input_dictionary['ca']
    thal= input_dictionary['thal']
  

    input_list = [age, sex, cp,trestbps, chol,fbs,restecg,thalach, exang,oldpeak, slope,ca,thal]
    
    prediction = heart_model.predict([input_list])
    if prediction[0] == 0:
        return 0
    
    else:
        return 1


    