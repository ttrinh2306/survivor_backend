import transformers
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, TFDistilBertModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import os

app = Flask(__name__)
CORS(app)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

wd = os.getcwd()
df = pd.read_excel(wd + '/input/top_players.xlsx', engine='openpyxl')

@app.route('/compare_bio', methods=['POST'])
def compare_bio():
    user_bio = request.json['bio']

    # Function to get BERT embeddings
    # def get_bert_embedding(text):
    #     inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True)
    #     outputs = model(inputs)
    #     return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    
     # Function to get BERT embeddings
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()   

    encoded_bio = get_bert_embedding(user_bio)

    # Vectorized computation of cosine similarity
    embeddings_list = [np.array(ast.literal_eval(x)).flatten() for x in df['embeddings']]
    similarity_scores = cosine_similarity([encoded_bio.flatten()], embeddings_list)[0]
    df['similarity'] = similarity_scores
    
    most_similar_player = df.loc[df['similarity'].idxmax(), 'name'].replace('_', ' ')

    return jsonify({"similar_player": most_similar_player})

if __name__ == '__main__':
    app.run(debug=True)