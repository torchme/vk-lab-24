# Предсказание CTR для Постов ВКонтакте

## Описание проекта

Этот проект направлен на разработку модели для предсказания Click-Through Rate (CTR) постов в социальной сети ВКонтакте. CTR определяется как вероятность конверсии просмотра поста в различные активности, такие как лайки и пересылки в личные сообщения. В проекте используются как текстовые, так и визуальные данные постов, а также числовые признаки. Для объединения предсказаний базовых моделей применяется CatBoost как мета-модель.

## Содержание

- [Описание проекта](#описание-проекта)
- [Содержание](#содержание)
- [Технологии](#технологии)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)


## Технологии

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- PyTorch
- Hugging Face Transformers
- Pillow


## Структура проекта

```
VKLAB/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train_text_model.py
│   ├── train_image_model.py
│   ├── train_catboost.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

* data/: Содержит исходный датасет.
* notebooks/: Jupyter Notebook для первичного разведывательного анализа (EDA).
* src/: Исходный код проекта.
    * data_preprocessing.py: Скрипты для предобработки данных.
    * models.py: Определение архитектур моделей.
    * train_text_model.py: Скрипт для обучения текстовой модели.
    * train_image_model.py: Скрипт для обучения визуальной модели.
    * train_catboost.py: Скрипт для обучения мета-модели CatBoost.
    * utils.py: Вспомогательные функции.
* requirements.txt: Список зависимостей проекта.
* README.md: Описание проекта.
* .gitignore: Файлы и папки, игнорируемые Git.

## Установка

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/your-username/vklab.git
   ```

2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
