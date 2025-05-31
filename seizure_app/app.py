import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("model/cnn_bilstm_epilepsy_model.h5")

# Global variable to store latest CSV
latest_csv = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global latest_csv

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = file.filename

    # Read and preprocess CSV
    data = pd.read_csv(file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    if 'y' in data.columns:
        data = data.drop(columns=['y'])

    # Preprocess for model input
    X = data.values.astype(np.float32)  # (10, 178)
    X = np.expand_dims(X, axis=-1)      # (10, 178, 1)

    # Predict
    preds = model.predict(X)
    if preds.shape[1] == 1:
        classes = (preds > 0.5).astype(int).flatten()
    else:
        classes = np.argmax(preds, axis=1)

    # Append results to DataFrame
    data['Prediction'] = classes
    seizure_count = np.sum(classes == 1)
    non_seizure_count = np.sum(classes == 0)

    # Save CSV to memory
    csv_buffer = BytesIO()
    data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    latest_csv = csv_buffer

    return render_template('result.html',
                           filename=filename,
                           seizure=seizure_count,
                           non_seizure=non_seizure_count,
                           total=len(classes))

@app.route('/download')
def download():
    global latest_csv
    if latest_csv:
        latest_csv.seek(0)  # Reset pointer just in case
        return send_file(latest_csv,
                         mimetype='text/csv',
                         download_name='seizure_predictions.csv',
                         as_attachment=True)
    else:
        return "No result available for download.", 404

if __name__ == '__main__':
    app.run(debug=True)
