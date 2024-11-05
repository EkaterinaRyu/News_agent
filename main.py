from newsapi.newsapi_client import NewsApiClient
from bs4 import BeautifulSoup
import requests
import io
import json
import os
from pathlib import Path
from docx import Document
from deep_translator import GoogleTranslator

from datetime import datetime, timedelta
from tqdm import tqdm
from PIL import Image

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from diffusers import DiffusionPipeline
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )

# получение ключей API из файла
with open("api_keys.json") as f:
    config = json.load(f)

NEWS_API_KEY = config['NEWS_API_KEY']
MISTRAL_API_KEY = config['MISTRAL_API_KEY']

# метод получения последних новостей по API из агрегатора
def get_news(NEWS_API_KEY: str = NEWS_API_KEY) -> list:
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # список запросов для фильтрации новостей
    queries_1 = ['moms on the go', 'working moms', 'stay-at-home moms', 
               'modern moms', 'mom hacks', 'mom guilt', 'mom burnout', 
               'mom self-care', 'mom tips', 'mom advice', 'mom support', 
               'mom community', 'mom fitness', 'mom nutrition', 'mom life balance', 
               'mom self-care', 'mom wellness', 'motherhood', 'Early education', 
               'Preschool education', 'Kindergarten education']

    query_string = ' OR '.join(queries_1)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    # агрегатор возвращает ссылки на релевантные стратьи и их заголовоки
    response = newsapi.get_everything(q=query_string, language='en', from_param=start_date, sort_by='relevancy')
    print(f"Total articles found: {response['totalResults']}")

    articles_list = []
    # статьи из агрегатора извлекаются на английском языке, 
    # для передачи в модель необходимо перевести на русский
    translator = GoogleTranslator(source='en', target='ru')

    # для каждой релевантной статьи необходимо сделать запрос и извлечь текст
    # для примера запросы делаются только для первых трёх статей
    for article in response['articles'][:3]:
        url = article['url']
        headers = {"accept": "application/json"}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                l = ''
                for paragraph in tqdm(paragraphs):
                    l += translator.translate(paragraph.text)
                articles_list.append(l)
            else:
                print(f"Request to {url} failed with status code {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Request to {url} timed out")
        except Exception as e:
            print(f"An error occurred: {e}")

    return articles_list

# метод получения последних статей со специализированного сайта
def get_mom_news() -> list:
    response = requests.get(url='https://libertymag.ru/novosti-mamy-deti')
    s = BeautifulSoup(response.text, 'html.parser')

    articles_list = []

    for article in s.find_all('article'):
        date = article.find('time')['datetime']

        # check if the article was posted in the last week
        publication_date = datetime.strptime(date.split('T')[0], '%Y-%m-%d')
        current_date = datetime.now()
        difference = current_date - publication_date
        if difference > timedelta(days=7):
            # articles are in reverce chronological order, 
            # so stopping the retrieval when old article occurs
            break

        url = article.h2.a['href']
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                h = soup.find('h1', attrs={'class': 'post-title'})
                a = soup.find('div', attrs={'class': 'inner-post-entry'})
                paragraphs = a.find_all('p')
                l = h.text + ' '
                for paragraph in tqdm(paragraphs):
                    l += paragraph.text + ' '
                articles_list.append(l)
            else:
                print(f"Request to {url} failed with status code {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Request to {url} timed out")
        except Exception as e:
            print(f"An error occurred: {e}")

    return articles_list

# метод получения сжатого текста с помощью запроса по API
def get_summary_api_mistral(text: str, prompt: str, length: int, 
                            MISTRAL_API_KEY: str = MISTRAL_API_KEY) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + MISTRAL_API_KEY,
        "Content-Type": "application/json"
    }
    # промпт позволяет управлять поведением модели 
    data = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", 
             "content": """Я веду блог в котором размещаю последние новости, 
             а ты мой помощник. Я не говорю по английски, пожалуйста, всегда
             отвечай мне на русском языке."""},
            {"role": "user", 
             "content": f"""{prompt} Будь краток, сократи следующий текст 
             до {length} слов, обязательно помести его между [], верни только 
             сокращенный текст указанной длинны в формате [сокращенный текст],
             это очень важно. Вот текст:\n\n{text}"""}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    summary = response.json()["choices"][0]["message"]["content"]
    # иногда модель помимо пересказа возвращает дополнительную информацию
    # указание в промпте просьбы поместить пересказ между квадратными скобками 
    # позволяет извлечь только необходимый текст
    summary = summary.split('[')[1].split(']')[0]

    return summary

# метод получения сжатого текста с помощью локальной модели
def get_summary_model(text: str, length: int) -> str:
    model = AutoModelForSeq2SeqLM.from_pretrained("d0rj/rut5-base-summ")
    tokenizer = AutoTokenizer.from_pretrained("d0rj/rut5-base-summ")

    inputs = tokenizer.encode("сделай краткий пересказ: " + text, 
                              return_tensors='pt', 
                              max_length=1024, 
                              truncation=True)
    outputs = model.generate(inputs, 
                             max_length=length+50, # ограничение длины генерации
                             min_length=length, 
                             length_penalty=2.0, 
                             num_beams=4, 
                             early_stopping=True, 
                             do_sample=True, 
                             temperature=0.7)
    
    summary = tokenizer.decode(outputs[0])

    return summary

# метод генерации изображения с помощью локальной модели
def generate_image_model(text: str) -> Image:
    pipe = DiffusionPipeline.from_pretrained("openskyml/lexica-aperture-v3-5")
    pipe = pipe.to(device)
    image = pipe(text).images[0]

    return image

# метод генерации изображения с помощью удаленной модели
def generate_image_api_pollinations(text: str) -> Image:
    try:
        url = f"https://image.pollinations.ai/prompt/{text}"
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            print(f"Request to {url} failed with status code {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"Request to {url} timed out")
    except Exception as e:
        print(f"An error occurred: {e}")
    
# метод агента
def news_agent() -> None:

    # создание папки на рабочем столе
    folder = Path.home() / "Desktop" / "blog_files"
    os.makedirs(folder, exist_ok=True)

    # получение статей
    # приоритетным является получение статей со специализированного сайта
    try:
        news_list = get_mom_news()
    except:
        print("Couldn't access mom's blog https://libertymag.ru/novosti-mamy-deti, retrieving news by API")
        news_list = get_news()
    
    for i, article in tqdm(enumerate(news_list)):
        # генерация текста для блога
        # приоритетным является запрос по API
        prompt = "Пожалуйста, помоги мне написать пост для моего блога."
        try:
            blog_post = get_summary_api_mistral(article, prompt, 200)
        except Exception as e:
            # print(e)
            print("Couldn't use Mistral to get summary, loading local model")
            blog_post = get_summary_model(article, 200)

        # генерация промпта для создания изображнеия
        # приоритетным является запрос по API
        prompt = "Пожалуйста, помоги мне написать промпт для генерации изображения к посту в моём блоге."
        try:
            image_prompt = get_summary_api_mistral(blog_post, prompt, 20)
        except Exception as e:
            # print(e)
            print("Couldn't use Mistral to get image prompt, loading local model")
            image_prompt = get_summary_model(blog_post, 20)
        # генерация изображения
        # приоритетным является запрос на сайт
        try:
            image = generate_image_api_pollinations(image_prompt)
        except Exception as e:
            print(e)
            print("Couldn't use Pollinations to generate image, loading local model")
            image = generate_image_model(image_prompt)

        # сохранение результатов
        doc = Document()
        paragraph = doc.add_paragraph(blog_post)
        doc.save(folder / f"blog_post_{i}.docx")

        with open(folder / f"image_prompt_{i}.txt", "w", encoding="utf-8") as f:
            f.write(image_prompt)
        
        image.save(folder / f"image_{i}.png")


news_agent()
