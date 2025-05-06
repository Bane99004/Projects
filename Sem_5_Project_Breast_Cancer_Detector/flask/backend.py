from flask import Flask,request,render_template,jsonify,make_response,session,redirect,flash,url_for
import jwt
from datetime import datetime, timedelta 
from functools import wraps
from step2 import train, predict
from step3 import train_2, predict_2
import os
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
symptoms ={
    1:"Do you have cracks on your skin?",
    2:"Do you feel any lumps?",
    3:"Do you have swelling?",
    4:"Do you have a rash or redness on skin?",
    5:"Is there discharge from your nipples from the following(blood, puss)?",
    6:"Do you have pain in any part of your body?"
}


train()
train_2()
@app.route('/')
def page():
  return render_template('home.html')
@app.route('/checkup',methods=['GET'])
def index():
    current_symptom = 1
    yes_count = 0
    return render_template('examine.html', question=symptoms[current_symptom])

@app.route('/update', methods=['GET'])
def update():

    current_symptom = int(request.args.get('current_symptom', 1))
    yes_count = int(request.args.get('yes_count', 0))#0 = default value ----args = get the values from arguments /update?args

    response = request.args.get('response', '').strip().lower() #fetch yes or no----strip removes whitespaces(%20)----lower ->lowercase
    if response == 'yes':
        current_symptom += 1
    elif response == 'no':
        current_symptom += 1

    if current_symptom >= len(symptoms):
        recommendation = "We recommend you to go to Step 2 for further examination of your breast to detect tumor.You can navigate through below link" if yes_count > 0 else "It seems like you're fine, but if symptoms persist or you want 100% assurance , please proceed to Step 2.<br>You can navigate through below link"
        return jsonify({'result': recommendation})
    question = symptoms[current_symptom]
    return jsonify({'question': question,
                    'current_symptom':current_symptom,
                    })
if not os.path.exists(app.config['UPLOAD_FOLDER']):
  os.makedirs(app.config['UPLOAD_FOLDER'])
@app.route('/external')
def index_2():
    return render_template('external.html')

@app.route('/upload',  methods =['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index_2'))
    file = request.files['image']
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash('File uploaded sucessfully')
    prediction = predict(filepath)
    return jsonify({
        'prediction': prediction
    })
@app.route('/step3')
def index_3():
    return render_template('step3.html')
@app.route('/upload2', methods=['POST'])
def upload2():
    if 'image' not in request.files:
        flash("No file part")
        return redirect(url_for('index_3'))
    file = request.files['image']
    if file:
        filename = file.filename 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash('File uploaded sucessfully')
    prediction = predict_2(filepath)
    return jsonify({
        'prediction':prediction
    })
if __name__ == '__main__':
  app.run(debug = False) 
  
  