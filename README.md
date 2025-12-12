# Создание и оптимизация ResNet18

https://github.com/physicorym/designing_neural_network_architectures_2025_01/tree/main/seminar_02



## Getting Started
These instructions will give you a copy of the project up and running on your local machine for development and testing 
purposes. There isn't much to do, just install the prerequisites and download all the files.

### Prerequisites
Create an environment into the folder `.venv`
```
python -m venv .venv
```

Activate the environment
```
.venv\Scripts\activate
```

Run the command:
```
pip install -r requirements.txt
```

## Download datasets
### tiny-imagenet-200
https://disk.yandex.ru/d/adWo9fVCLuVQ0Q


```
pip freeze > requirements.txt
```



## Usage
```
python src/main.py --hypes src\hyperparameters\train.json
```
- `--hypes`, path to configuration file.