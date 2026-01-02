# Создание и оптимизация ResNet18

https://github.com/physicorym/designing_neural_network_architectures_2025_01/tree/main/seminar_02

## Архитектуры моделей

### Архитектура базовая: `[2, 2, 2, 2]`
- `"layers_num": 4`
- `"block_size": 2`

| Layer / Operation                                  | Shape / Size      |
|----------------------------------------------------|-------------------|
| Input                                              | `(B, 3, 64, 64)`  |
| `Conv2d(3→32, kernel_size=7, stride=2, padding=3)` | `(B, 32, 32, 32)` |
| `BatchNorm2d(32)`                                  | `(B, 32, 32, 32)` |
| `activation`                                       | `(B, 32, 32, 32)` |
| `MaxPool2d(kernel_size=3, stride=1, padding=1)`    | `(B, 32, 32, 32)` |
| `Layer0`: 2× `BasicBlock` (32 channels, stride=2)  | `(B, 32, 16, 16)` |
| `Layer1`: 2× `BasicBlock` (64 channels, stride=2)  | `(B, 64, 8, 8)`   |
| `Layer2`: 2× `BasicBlock` (128 channels, stride=2) | `(B, 128, 4, 4)`  |
| `Layer3`: 2× `BasicBlock` (256 channels, stride=2) | `(B, 256, 2, 2)`  |
| `AdaptiveAvgPool2d(output_size=(1, 1))`            | `(B, 256, 1, 1)`  |
| `Flatten()`                                        | `(B, 256)`        |
| `Linear(256→ 10)`                                  | `(B, 10)`         |
| Output                                             | `(B, 10)`         |

### Архитектура с меньшим количеством слоев: `[2, 2, 2]`
- `"layers_num": 3`
- `"block_size": 2`

| Layer / Operation                                  | Shape / Size      |
|----------------------------------------------------|-------------------|
| Input                                              | `(B, 3, 64, 64)`  |
| `Conv2d(3→64, kernel_size=7, stride=2, padding=3)` | `(B, 64, 32, 32)` |
| `BatchNorm2d(64)`                                  | `(B, 64, 32, 32)` |
| `activation`                                       | `(B, 64, 32, 32)` |
| `MaxPool2d(kernel_size=3, stride=1, padding=1)`    | `(B, 64, 32, 32)` |
| `Layer0`: 2× `BasicBlock` (64 channels, stride=2)  | `(B, 64, 16, 16)` |
| `Layer1`: 2× `BasicBlock` (128 channels, stride=2) | `(B, 128, 8, 8)`  |
| `Layer2`: 2× `BasicBlock` (256 channels, stride=2) | `(B, 256, 4, 4)`  |
| `AdaptiveAvgPool2d(output_size=(1, 1))`            | `(B, 256, 1, 1)`  |
| `Flatten()`                                        | `(B, 256)`        |
| `Linear(256→10)`                                   | `(B, 10)`         |
| Output                                             | `(B, 10)`         |

### Архитектура с увеличенной глубиной слоев `[3, 3, 3, 3]`
- `"layers_num": 4`
- `"block_size": 3`

| Layer / Operation                                  | Shape / Size      |
|----------------------------------------------------|-------------------|
| Input                                              | `(B, 3, 64, 64)`  |
| `Conv2d(3→32, kernel_size=7, stride=2, padding=3)` | `(B, 32, 32, 32)` |
| `BatchNorm2d(32)`                                  | `(B, 32, 32, 32)` |
| `activation`                                       | `(B, 32, 32, 32)` |
| `MaxPool2d(kernel_size=3, stride=1, padding=1)`    | `(B, 32, 32, 32)` |
| `Layer0`: 3× `BasicBlock` (32 channels, stride=2)  | `(B, 32, 16, 16)` |
| `Layer1`: 3× `BasicBlock` (64 channels, stride=2)  | `(B, 64, 8, 8)`   |
| `Layer2`: 3× `BasicBlock` (128 channels, stride=2) | `(B, 128, 4, 4)`  |
| `Layer3`: 3× `BasicBlock` (256 channels, stride=2) | `(B, 256, 2, 2)`  |
| `AdaptiveAvgPool2d(output_size=(1, 1))`            | `(B, 256, 1, 1)`  |
| `Flatten()`                                        | `(B, 256)`        |
| `Linear(256→10)`                                   | `(B, 10)`         |
| Output                                             | `(B, 10)`         |

### Архитектура с уменьшенной глубиной слоев `[1, 1, 1, 1]`
- `"layers_num": 4`
- `"block_size": 1`

| Layer / Operation                                  | Shape / Size      |
|----------------------------------------------------|-------------------|
| Input                                              | `(B, 3, 64, 64)`  |
| `Conv2d(3→32, kernel_size=7, stride=2, padding=3)` | `(B, 32, 32, 32)` |
| `BatchNorm2d(32)`                                  | `(B, 32, 32, 32)` |
| `activation`                                       | `(B, 32, 32, 32)` |
| `MaxPool2d(kernel_size=3, stride=1, padding=1)`    | `(B, 32, 32, 32)` |
| `Layer0`: 1× `BasicBlock` (32 channels, stride=2)  | `(B, 32, 16, 16)` |
| `Layer1`: 1× `BasicBlock` (64 channels, stride=2)  | `(B, 64, 8, 8)`   |
| `Layer2`: 1× `BasicBlock` (128 channels, stride=2) | `(B, 128, 4, 4)`  |
| `Layer3`: 1× `BasicBlock` (256 channels, stride=2) | `(B, 256, 2, 2)`  |
| `AdaptiveAvgPool2d(output_size=(1, 1))`            | `(B, 256, 1, 1)`  |
| `Flatten()`                                        | `(B, 256)`        |
| `Linear(256→10)`                                   | `(B, 10)`         |
| Output                                             | `(B, 10)`         |

### Суммарное количество обучаемых параметров

| Конфигурация        | Параметры  |
|---------------------|------------|
| 4x1: `[1, 1, 1, 1]` | 1 233 898  |
| 4x2: `[2, 2, 2, 2]` | 2 802 538  |
| 3x2: `[2, 2, 2]`    | 2 789 578  |
| 4x3: `[3, 3, 3, 3]` | 4 371 178  |


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
python src/main.py --hypes src\hyperparameters\config.json
```
- `--hypes`, path to configuration file.

`save_policy`: "all", "best"
"early_stop" if "early_stop_number" <= 0