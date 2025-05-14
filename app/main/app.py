from flask import Flask, render_template, request
from app.models.inference import recommendation_system

app = Flask(__name__, template_folder='../static/templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    city = str(request.form['city'])
    catg = str(request.form['category'])
    desc = request.form['description']

    results = recommendation_system(description=desc, category=catg, city=city)
    return render_template('result.html', results=results)