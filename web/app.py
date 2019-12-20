from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    return render_template('search.html', movies=find(request.form['statement']))


def find(statement):
    return ['映画1',
            '映画2',
            '映画3']


if __name__ == '__main__':
    app.run()
