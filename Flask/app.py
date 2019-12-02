from flask import Flask, escape, request,render_template

app = Flask(__name__)

@app.route('/index')
def index():
    return render_template("index.html")

   