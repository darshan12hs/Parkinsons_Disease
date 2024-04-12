from flask import Flask, render_template, request
import joblib

# load the model
model=joblib.load("model_EVC.pkl")

# Initialize the app
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process',methods=['post'])
def form_data():
    MDVP_Fo=request.form.get("MDVP:Fo(Hz)")
    MDVP_Fhi=request.form.get("MDVP:Fhi(Hz)")
    MDVP_Flo=request.form.get("MDVP:Flo(Hz)")
    MDVP_Jitter_=request.form.get("MDVP:Jitter(%)")
    MDVP_Jitter_Abs=request.form.get("MDVP:Jitter(Abs)")
    MDVP_RAP=request.form.get("MDVP:RAP")
    MDVP_PPQ=request.form.get("MDVP:PPQ")
    Jitter_DDP=request.form.get("Jitter:DDP")
    MDVP_Shimmer=request.form.get("MDVP:Shimmer")
    MDVP_Shimmer_dB=request.form.get("MDVP:Shimmer(dB)")
    Shimmer_APQ3=request.form.get("Shimmer:APQ3")
    Shimmer_APQ5=request.form.get("Shimmer:APQ5")
    MDVP_APQ=request.form.get("MDVP:APQ")
    Shimmer_DDA=request.form.get("Shimmer:DDA")
    NHR=request.form.get("NHR")
    HNR=request.form.get("HNR")
    RPDE=request.form.get("RPDE")
    DFA=request.form.get("DFA")
    spread1=request.form.get("spread1")
    spread2=request.form.get("spread2")
    D2=request.form.get("D2")
    PPE=request.form.get("PPE")
    
    result=model.predict([[float(MDVP_Fo), float(MDVP_Fhi), float(MDVP_Flo), float(MDVP_Jitter_),
       float(MDVP_Jitter_Abs), float(MDVP_RAP), float(MDVP_PPQ), float(Jitter_DDP),
       float(MDVP_Shimmer), float(MDVP_Shimmer_dB), float(Shimmer_APQ3), float(Shimmer_APQ5),
       float(MDVP_APQ), float(Shimmer_DDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1),
       float(spread2), float(D2), float(PPE)]])
    
    if result==1:
        data="Parkinson's disease is detected in the individual"
    else:
        data="The Individual is doing fine & doesn't have parkinson's disease"
    
    return data

# run the application
app.run('0.0.0.0',port='8090')
    
