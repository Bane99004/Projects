from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    input_value = data['inputValue']
    # Process the input value here
    # print(f"Received value: {input_value}")
    return jsonify({'input_value': input_value})

if __name__ == '__main__':
    app.run(debug=True)
