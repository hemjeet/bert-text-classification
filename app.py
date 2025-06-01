import streamlit as st
import os
import boto3
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---- CONFIGURE THESE ----
S3_BUCKET = 'kumar-buckets'
S3_MODEL_DIR = 'bert_finetuned_model/'  # prefix in S3
LOCAL_MODEL_DIR = 'bert_finetuned_model'

# --------------------------

def download_model_from_s3():
    s3 = boto3.client(
        service_name= 's3',
        region_name= 'us-east-1',
        aws_access_key_id= os.getenv('aws_access_key_id'),
        aws_secret_access_key= os.getenv('aws_secret_access_key')
        )
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    objects = s3.list_objects_v2(Bucket = S3_BUCKET, Prefix = S3_MODEL_DIR)
    for obj in objects.get('Contents', []):
        # Skip the folder itself
        if obj['Key'][-1] == '/':
            continue
            
        local_file_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(obj['Key']))
        with st.spinner(f'Downloading {obj}...'):
            s3.download_file(S3_BUCKET, obj['Key'], local_file_path)
        print(f"Downloaded {obj['Key']} to {local_file_path}")
    st.success("Model loaded successfully!")

@st.cache_resource
def load_model():
    download_model_from_s3()
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    model.eval()
    return tokenizer, model

# --- Streamlit UI ---
st.title("Sentiment Analysis with BERT")
tokenizer, model = load_model()

user_input = st.text_area("Enter a review", "The movie was awesome!")

if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1).squeeze().tolist()

    label_map = {0: "Negative", 1: "Positive"}
    st.write(f"**Prediction:** {label_map[prediction]}")
    st.write(f"**Confidence:** {prob[prediction]:.2%}")
