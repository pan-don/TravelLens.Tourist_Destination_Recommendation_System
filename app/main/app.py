from flask import Flask, render_template, request
from app.models.recommender import recommender_system

app = Flask(__name__, template_folder='../static/templates')

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman hasil rekomendasi
@app.route('/result', methods=['POST'])
def result():
    city = int(request.form['city'])
    catg = int(request.form['category'])
    desc = request.form['description']

    results = recommender_system(input_description=desc, input_category=catg, input_city=city)

    return render_template('result.html', results=results)