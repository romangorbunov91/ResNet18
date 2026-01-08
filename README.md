# Создание и оптимизация ResNet18
Поэтапная разработка кастомной ResNet18 модели-классификатора с анализом влияния различных архитектурных решений на производительность.

## Часть 1: Подготовка данных
Создан датакласс [TinyImageNetDataset.py](src\datasets\TinyImageNetDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и аннотациям, загрузка тренировочного и валидационного датасетов по выбранным классам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: загрузка и возврат одного примера (изображение + метка).

## Часть 2: Базовая архитектура ResNet18

В [model_structure.py](src\models\model_structure.py) реализован `class customResNet18` с возможностью инициализации архитектуры модели под следующие входные параметры:
- `num_classes` - количество классов на выходе; например, `num_classes=10`;
- `layers_config` - слои модели в формате списка; например, `[2, 2, 2, 2]` - `"layers_num": 4`, `"block_size": 2`;
- `activation` - функция активации (`ReLU`, `LeakyReLU`, `ELU`, или `GELU`);
- `in_channels` - количество входных каналов; например, для RGB-картинок `in_channels=3`;
- `layer0_channels` - количество каналов на входе первого базового слоя.

### 2.1. Реализация Basic Block

В [model_structure.py](src\models\model_structure.py) реализован `class BasicBlock` с выбором функции активации в слое `activation` при инициализации.

```
Input →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  →
  ↓                                                                     
Conv2d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)      ↓
  ↓                                                                     
BatchNorm2d                                                                 ↓
  ↓                                                                     
activation (ReLU, LeakyReLU, ELU, or GELU)                                  ↓
  ↓
Conv2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)           ↓
  ↓
BatchNorm2d                                                                 ↓
  ↓
  + ← Skip Connection  ←  ←  ←  ←  ←  ← downsample  ←  ←  ←  ←  ←  ←  ←  ←  ←
  ↓
activation (ReLU, LeakyReLU, ELU, or GELU)
  ↓
Output
```
### 2.2. Реализация ResNet18

#### Архитектура базовая (baseline): `[2, 2, 2, 2]`
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

### 2.3. Ограничения для базовой модели
- Общее количество параметров - не более 5 миллионов: ✅ **2802538**.
- Максимальное количество каналов - до 512: ✅ **256**.

### 2.4. Скрипт обучения

#### Конфигурирование проекта
Гиперпараметры задаются в файле [config.json](src\hyperparameters\config.json), включая:
- архитектура модели: `layers_num`, `block_size`, `activation`;
- выбранные классы датасета: `selected_classes`;
- параметры обучения: `epochs`, `batch_size`, `solver`;
- политика обучения: `save_policy` - "all", "best" (политика "early_stop" выбирается установкой параметра "early_stop_number" > 0).

#### Обучение
Обучение реализовано в [train.py](src\train.py) в виде класса `ResNet18Trainer` со следующими методами:
- `__init__` - инициализация переменных класса в соответствии с гиперпараметрами из файла конфигурации проекта;
- `init_model` - установка функции ошибки, инициализация/загрузка модели, загрузка датасета;
- `__train` - обучение по батчам;
- `__val` - валидация по батчам;
- `train` - основной цикл обучения/валидации по эпохам;
- `update_metrics` - аккумулирование losses/accuracy посредством [average_meter.py](src\utils\average_meter.py).

Рекомендуется работать с моделью посредством [main.py](src\main.py).

```
python src/main.py --hypes src\hyperparameters\config.json
```

**Запуск на обучение**
```
python src\main.py --hypes src\hyperparameters\config.json 
```

**Запуск на дообучение**
```
python src\main.py --hypes src\hyperparameters\config.json --resume checkpoints\tiny-imagenet-200\best_train_customResNet18.pth
```

**Запуск на тест**
```
python src\main.py --hypes src\hyperparameters\config.json --resume checkpoints\tiny-imagenet-200\best_train_customResNet18.pth --phase test
```

Логи обучения хранятся в [train_logs](train_logs).

#### 2.5: Визуализация базовых результатов

Графики построены в [main_notebook.ipynb](main_notebook.ipynb).

<p align="center" width="100%">
  <img src="./readme_img/loss_acc_4x2_ReLU_Adam.png"
  style="background-color: white; padding: 0;
  width="100%" />
</p>

### Выводы по базовой модели

- После 10й эпохи обучения качество на валидации не меняется, модель переходит в переобучение.
- Достигается точность на валидации около 55%.

## Часть 3: Поэтапная оптимизация модели
### 3.1: Оптимизация количества каналов
#### Архитектура с меньшим количеством слоев: `[2, 2, 2]`
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


### 3.2: Эксперименты с количеством residual блоков

#### Архитектура с увеличенной глубиной слоев `[3, 3, 3, 3]`
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

#### Архитектура с уменьшенной глубиной слоев `[1, 1, 1, 1]`
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

### 3.3: Эксперименты с функциями активации



## Часть 4: Финальная модель и тестирование

### Суммарное количество обучаемых параметров

| Конфигурация        | Параметры  |
|---------------------|------------|
| 4x1: `[1, 1, 1, 1]` | 1 233 898  |
| 4x2: `[2, 2, 2, 2]` | 2 802 538  |
| 3x2: `[2, 2, 2]`    | 2 789 578  |
| 4x3: `[3, 3, 3, 3]` | 4 371 178  |

### 4.4: Сравнительная таблица всех экспериментов

Создайте итоговую таблицу со всеми результатами:

| Этап          | Конфигурация          | Параметры | Val Accuracy  | Train Accuracy    |
|---------------|-----------------------|-----------|---------------|-------------------|
| **Baseline**  | [2,2,2,2]             | X.XM      | XX.X%         | XX.X%             |
| **3.1-A**     | [2,2,2,2]             | X.XM      | XX.X%         | XX.X%             |
| **3.1-B**     | [2,2,2]               | X.XM      | XX.X%         | XX.X%             |
| **3.2-A**     | [1,1,1,1]             | X.XM      | XX.X%         | XX.X%             |
| **3.2-B**     | [2,2,2,2]             | X.XM      | XX.X%         | XX.X%             |
| **3.2-C**     | [3,3,3,3]             | X.XM      | XX.X%         | XX.X%             |
| **3.3-A**     | [2,2,2,2] ReLU        | X.XM      | XX.X%         | XX.X%             |
| **3.3-B**     | [2,2,2,2] LeakyReLU   | X.XM      | XX.X%         | XX.X%             |
| **3.3-C**     | [2,2,2,2] ELU         | X.XM      | XX.X%         | XX.X%             |
| **3.3-D**     | [2,2,2,2] GELU        | X.XM      | XX.X%         | XX.X%             |
| **Final**     | [2,2,2,2] ReLU        | X.XM      | XX.X%         | XX.X%             |

# Выводы
- Лучший результат показала конфигурация `[2, 2, 2, 2] ReLU`.
- Есть ли признаки переобучения (большая разница между train и val)?


Визуализация через https://netron.app/

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
см. раздел

## Reference
- [Полный текст задания](https://github.com/physicorym/designing_neural_network_architectures_2025_01/tree/main/seminar_02)