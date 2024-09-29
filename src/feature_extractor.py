import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path

        # Инициализация моделей
        self._initialize_models()

    def _initialize_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.bert_model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased').to(self.device)
        self.bert_model.eval()

        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.resnet.eval()
        self.resnet_features = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Средние значения ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_data(self, data):
        data['text'] = data['text'].fillna("")
        assert data['text'].isnull().sum() == 0, "Пропущенные значения в 'text' после заполнения"
        return data

    def get_embeddings(self, data):
        bert_embeddings = self._get_bert_embeddings(data['text'])
        image_features = self._get_image_features(data['photo'])
        tfidf_embeddings = self._get_tfidf_embeddings(data['text'])
        text_features_df = self.extract_text_features(data)
        return bert_embeddings, image_features, tfidf_embeddings, text_features_df

    def _get_bert_embeddings(self, texts):
        embeddings_file = os.path.join(self.data_path, 'embeddings', 'bert_embeddings.npy')
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
        else:
            embeddings = self._calculate_bert_embeddings(texts)
            os.makedirs(os.path.join(self.data_path, 'embeddings'), exist_ok=True)
            np.save(embeddings_file, embeddings)
        return embeddings

    def _calculate_bert_embeddings(self, texts):
        embeddings = []
        for text in tqdm(texts, desc="Извлечение BERT эмбеддингов"):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return embeddings

    def _get_image_features(self, images):
        embeddings_file = os.path.join(self.data_path, 'embeddings', 'resnet_embeddings.npy')
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
        else:
            embeddings = self._calculate_image_features(images)
            os.makedirs(os.path.join(self.data_path, 'embeddings'), exist_ok=True)
            np.save(embeddings_file, embeddings)
        return embeddings

    def _calculate_image_features(self, images):
        embeddings = []
        for img in tqdm(images, desc="Извлечение признаков изображений"):
            if pd.isnull(img) or img == '':
                features = np.zeros(2048)
            else:
                features = self._process_image(img)
            embeddings.append(features)
        embeddings = np.array(embeddings)
        return embeddings

    def _process_image(self, img):
        try:
            image = Image.open(BytesIO(base64.b64decode(img))).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.resnet_features(image)
            features = features.squeeze().cpu().numpy()
        except Exception:
            features = np.zeros(2048)
        return features

    def extract_text_features(self, data):
        # Предполагается, что 'text' уже заполнен и не содержит NaN
        features_list = data['text'].apply(self._extract_text_features).tolist()
        text_features_df = pd.DataFrame(features_list)
        return text_features_df

    def _extract_text_features(self, text):
        features = {}
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = sum(len(word) for word in text.split())
        features['sentence_count'] = text.count('.')
        features['avg_word_length'] = features['char_count'] / features['word_count'] if features['word_count'] > 0 else 0
        features['link_count'] = len(re.findall(r'http[s]?://', text))
        features['has_link'] = 1 if features['link_count'] > 0 else 0
        features['special_char_count'] = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        features['uppercase_count'] = sum(1 for char in text if char.isupper())
        features['has_uppercase'] = 1 if features['uppercase_count'] > 0 else 0
        return features

    def _get_tfidf_embeddings(self, texts):
        tfidf_file = os.path.join(self.data_path, 'embeddings', 'tfidf_embeddings.npy')
        if os.path.exists(tfidf_file):
            embeddings = np.load(tfidf_file)
        else:
            embeddings = self._calculate_tfidf_embeddings(texts)
            os.makedirs(os.path.join(self.data_path, 'embeddings'), exist_ok=True)
            np.save(tfidf_file, embeddings)
        return embeddings

    def _calculate_tfidf_embeddings(self, texts):
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        embeddings = tfidf_vectorizer.fit_transform(texts).toarray()
        return embeddings
