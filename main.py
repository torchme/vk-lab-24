import torch
import pandas as pd
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.model import CatBoostModel
from src.trainer import Trainer

def main():
    # Путь к данным и моделям
    data_path = 'data'
    model_path = 'models/catboost_model'

    # Инициализация устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Инициализация классов
    data_loader = DataLoader(data_path)
    data = data_loader.load_data()
    
    activities = ['like', 'comment', 'hide', 'expand', 'open_photo', 'share_to_message', 'open']
    base_features = ['view', 'like', 'comment', 'hide', 'expand', 'open_photo', 'open', 'share_to_message', 'text', 'photo', 'engagement', 'engagement_conversion']

    # Проверка на наличие обработанных данных
    processed_data = data_loader.load_processed_data()
    if processed_data is not None:
        data_combined = processed_data
    else:
        feature_extractor = FeatureExtractor(device, data_path)
        data = feature_extractor.preprocess_data(data)

        # Расчет engagement и коэффициентов конверсии
        data['engagement'] = data['like'] + data['comment'] + data['hide'] + data['expand'] + data['open_photo'] + data['share_to_message'] + data['open']
        data['engagement_conversion'] = data['engagement'] / data['view']
        data['engagement_conversion'].fillna(0, inplace=True)

        # Расчет индивидуальных коэффициентов конверсии

        for activity in activities:
            data[f'{activity}_conversion'] = data[activity] / data['view']
            data[f'{activity}_conversion'].fillna(0, inplace=True)

        # Извлечение признаков
        bert_embeddings, image_features, tfidf_embeddings, text_features_df = feature_extractor.get_embeddings(data)

        # Объединение признаков
        bert_embeddings_2d = bert_embeddings.reshape(bert_embeddings.shape[0], bert_embeddings.shape[2])
        bert_df = pd.DataFrame(bert_embeddings_2d, columns=[f'bert_{i}' for i in range(bert_embeddings_2d.shape[1])])
        image_df = pd.DataFrame(image_features, columns=[f'image_{i}' for i in range(image_features.shape[1])])
        tfidf_df = pd.DataFrame(tfidf_embeddings, columns=[f'tfidf_{i}' for i in range(tfidf_embeddings.shape[1])])

        data_combined = pd.concat([
            data.reset_index(drop=True),
            bert_df.reset_index(drop=True),
            image_df.reset_index(drop=True),
            text_features_df.reset_index(drop=True),
            tfidf_df.reset_index(drop=True)
        ], axis=1)

        # Сохранение обработанных данных в Parquet
        data_loader.save_processed_data(data_combined)

    # Подготовка признаков и целевых переменных
    targets = [f'{activity}_conversion' for activity in activities]
    features = [col for col in data_combined.columns if col not in base_features + targets]
    targets = targets + ['engagement_conversion']

    X = data_combined[features].values
    y = data_combined[targets].values  # Мультивыход

    # Инициализация модели и тренера
    model = CatBoostModel(model_path)
    trainer = Trainer(model, data_path)

    # Обучение модели
    metrics = trainer.train(X, y, targets)

    # Вывод метрик
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

if __name__ == '__main__':
    main()