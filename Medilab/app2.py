from flask import Flask, request, render_template, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import pickle

# Flask app
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'harshal@1438'

db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)  # Use hashed passwords in production

# Patient Model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    mo_no = db.Column(db.String(15), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Patient {self.first_name} {self.last_name}>'

@app.before_request
def require_login():
    allowed_routes = ['login', 'register']
    if 'user_id' not in session and request.endpoint not in allowed_routes:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # Hash this in production

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()  # Hash comparison in production
        if user:
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        first_name = request.form['firstName']
        last_name = request.form['lastName']
        mo_no = request.form['moNo']
        age = request.form['age']
        location = request.form['location']

        new_patient = Patient(first_name=first_name, last_name=last_name,
                              mo_no=mo_no, age=age, location=location)
        
        try:
            db.session.add(new_patient)
            db.session.commit()
            flash('Patient details saved successfully!', 'success')
            return redirect(url_for('medilab'))
        except Exception as e:
            db.session.rollback()
            flash('Error saving patient details.', 'error')

    return render_template('index.html')

@app.route('/medilab')
def medilab():
    return render_template('medilab.html')

# Load dataset
sym_des = pd.read_csv("Medilab/dataset/symtoms_df.csv")
precautions = pd.read_csv("Medilab/dataset/precautions_df.csv")
workout = pd.read_csv("Medilab/dataset/workout_df.csv")
description = pd.read_csv("Medilab/dataset/description.csv")
medications = pd.read_csv('Medilab/dataset/medications.csv')
diets = pd.read_csv("Medilab/dataset/diets.csv")

# Load model
svc = pickle.load(open('Medilab/Model/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    filtered_desc = description[description['Disease'] == dis]['Description']
    if not filtered_desc.empty:
         desc = filtered_desc.values[0]
    else:
    # Handle the case when there is no matching disease
         desc = "No description available"

   # desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values[0] if pd.notnull(col)]
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    return desc, pre, med, die, wrkout

# Symptoms and diseases dictionaries
symptoms_dict = { ... }  # Add your existing symptoms_dict here
diseases_list = { ... }  # Add your existing diseases_list here

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms" or not symptoms.strip():
            message = "Please provide valid symptoms."
            return render_template('index.html', message=message)
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, my_diet, workout = helper(predicted_disease)

            return render_template('predict.html', predicted_disease=predicted_disease,
                                   dis_des=dis_des, my_precautions=precautions,
                                   my_medications=medications, my_diet=my_diet,
                                   my_workout=workout)

    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)
