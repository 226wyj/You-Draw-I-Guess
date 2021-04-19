from flask import Flask
from flask_bootstrap import Bootstrap
from flask import render_template

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)