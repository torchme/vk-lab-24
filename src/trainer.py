import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import wandb
import pandas as pd

class Trainer:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path

    def train(self, X, y, activities):
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Сохранение для повторного использования
        np.save(os.path.join(self.data_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.data_path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.data_path, 'y_test.npy'), y_test)

        # Инициализация WandB
        wandb.init(project='vklab_engagement_prediction', config=self.model.params)

        # Обучение модели
        self.model.train(X_train, y_train)

        # Предсказание и оценка
        y_pred = self.model.predict(X_test)

        # Сохранение предиктов
        predictions_file = os.path.join(self.data_path, 'predictions.npy')
        np.save(predictions_file, y_pred)

        metrics = []
        for i, activity in enumerate(activities):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            metrics.append({
                'Activity': activity,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            })
            # Логирование метрик и предиктов в WandB
            wandb.log({
                f'{activity}_MSE': mse,
                f'{activity}_RMSE': rmse,
                f'{activity}_R2': r2
            })

            # Логирование графиков предиктов vs фактических значений
            wandb.log({
                f'{activity}_predictions': wandb.Table(dataframe=pd.DataFrame({
                    'y_true': y_test[:, i],
                    'y_pred': y_pred[:, i]
                }))
            })

        # Сохранение модели
        self.model.save_model()

        wandb.finish()

        return metrics
