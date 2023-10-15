# <p align="center"> ЦИФРОВОЙ ПРОРЫВ: СЕЗОН ИИ </p>
# <p align="center"> Распознавание номеров железнодорожных вагонов </p>
<p align="center">
<img width="400" height="400" alt="photo" src="https://github.com/ankkarp/wagon-number-ocr/blob/kinowari/-removebg-preview.png">

</p> 


## Оглавление
1. [Задание](#1)
2. [Решение](#2)
3. [Запуск кода](#3)
4. [Уникальность нашего решения](#4)
5. [Стек](#5)
6. [Команда](#6)
7. [Ссылки](#7)

## <a name="1"> Задание </a>

В данном соревновании СТАРТЕРА предлагает разработать систему на основе фотографий с камер, установленных на различном расстоянии от объекта, участникам хакатона предстоит с применением технологий искусственного интеллекта создать программный модуль поиска и распознавания найденных на фотографии номеров железнодорожных вагонов, нанесенных на борт и шасси (платформу) с проверкой корректности распознавания по алгоритму создания номеров вагонов и передачей результатов по API. Использование системы распознавания номеров железнодорожных вагонов в системах динамического взвешивания позволяет исключить оператора из процесса взвешивания и автоматизировать бизнес-процессы предприятия. Решение кейса представляет собой программный код системы распознавания номеров вагонов для его встраивания в систему динамического взвешивания.

## <a name="2">Решение </a>

Решение представляет из себя последовательное применение моделей детекции и распознавания текста с последующей обработкой результата.

| До обработки  | После обработки |
| ------------- | ------------- |
| <img width="600" height="300" alt="image" src="https://github.com/ankkarp/wagon-number-ocr/blob/kinowari/photo_2023-10-15_00-12-44(%D0%B4%D0%BE).jpg">  | <img width="600" height="300" alt="image" src="https://github.com/ankkarp/wagon-number-ocr/blob/kinowari/photo_2023-10-15_00-02-06.jpg">  |


## <a name="3">Запуск кода </a>

### Последовательные шаги для запуска кода:
1. Склонируйте гит репозиторий;
```Bash
git clone https://github.com/ankkarp/wagon-number-ocr.git
```
2. Скачайте веса для модели детекции [yolov8.pt](https://drive.google.com/file/d/1_GgjGP_vOUZLzOk44dhArin81sYBRJI8/view?usp=drive_link);
3. Скачайте веса для модели выравнивания [moran.pth](https://drive.google.com/file/d/1hCFVOzW8J6l59G3jsYwAFmZyjs0XwpT6/view?usp=drive_link);
4. Установить pytorch под версию cuda: https://pytorch.org/get-started/locally/;
5. Установить:
 ```Bash
pip install -U openmim
pip install chardet
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmocr
pip install -U --force-reinstall charset-normalizer
```
5. Установить зависимости проекта:
 ```Bash
pip install -r requirements.txt
```
6. Запустите в командной строке следующую команду, предварительно заменив пути к папкам на ваши
```Bash
cd путь_до_папки_с_кодом_репозитория
python test.py -d "Путь до весов скаченной модели детекции" -i "путь до папки с фотографиями" -o 'название_файла_с_результатом.csv' -r 'путь_до весов скаченной модели выравнивания'
```
## <a name="4">Уникальность нашего решения </a>

Мы используем комбинацию моделей для распознавания цифр с последующей постобработкой в виде эвристик связанных с количеством предсказанных цифр

## <a name="5">Стек </a>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original-wordmark.svg" title="Python" alt="Puthon" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/pytorch/pytorch-original.svg" title="Pytorch" alt="Puthon" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Puthon" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="Puthon" width="40" height="40"/>&nbsp;
  <img src="https://github.com/gaotongxiao/mmocr/blob/0cd2878b048cacc85306ef02a5cb60a61de7f91b/resources/mmocr-logo.png" title="MMocr" alt="Puthon" width="60" height="40"/>&nbsp;
  <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" title="Modelscope" alt="Puthon" width="120" height="40"/>&nbsp;
  <img src="https://github.com/gradio-app/gradio/blob/main/readme_files/gradio.svg" title="Gradio" alt="Puthon" width="100" height="40"/>&nbsp;
  <img src="https://pjreddie.com/media/image/yologo_2.png" title="Yolo" alt="Puthon" width="100" height="40"/>&nbsp;
## <a name="6">Команда </a>


*Состав команды "Герои ML и Магии"*   
*Анна Карпова (https://github.com/ankkarp) - ML-engineer*    
*Александра Куроедова (https://github.com/c-nemo) - ML-engineer*  
*Анастасия Хан (https://github.com/Nanochka1) - Disigner*  
*Олег Сивец (https://github.com/OlegSivets) - ML-engineer*   
*Рената Аюпова (https://github.com/kinowari) - ML-engineer* 

## <a name="7">Ссылки </a>
[Гугл диск с материалами](https://drive.google.com/drive/u/0/folders/13MgumU4OoE917fjG94GmjqmIzjyqc-jl)
[ссылка на весы модели детекции](https://drive.google.com/file/d/1_GgjGP_vOUZLzOk44dhArin81sYBRJI8/view?usp=drive_link)
[ссылка на скринкаст](https://drive.google.com/file/d/1Wdu8nEqs_M4TL1Mfy7-lKL54340miua6/view?usp=drive_link)
