from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def probe(path):
    return jsonify({'status': 'ok', 'runtime': 'python-flask', 'probe': True}), 200
