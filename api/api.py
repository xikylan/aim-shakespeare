from flask import Flask, jsonify
from predict import predict

app = Flask(__name__)


@app.route('/api/tr/<text>')
def translate(text):
    output = predict(text)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=False)
