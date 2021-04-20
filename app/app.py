from flask import Flask
from flask_bootstrap import Bootstrap
from flask import render_template
# from flask_moment import Moment
from flask_wtf import Form

app = Flask(__name__)
bootstrap = Bootstrap(app)
# moment = Moment(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    return render_template('test.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(debug=True)