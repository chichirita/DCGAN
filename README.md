# DCGAN

Этот проект посвящен исследованию возможностей генеративно-состязательных сетей DCGAN в синтезе визуально сложных и разнородных данных. Модель обучена на комбинированном датасете, включающем архитектурные строения, природные ландшафты и логотипы футбольных клубов.

## Основные особенности

* **Стабильная архитектура**: Реализованы модули Generator и Discriminator с использованием транспонированных сверток и BatchNorm для предотвращения коллапса моды.
* **Оптимизированный пайплайн**: Использование Adam, функции активации LeakyReLU и нормализации данных в диапазоне .
* **Оптимизация обучения**: Полная поддержка NVIDIA CUDA для высокоскоростного обучения на GPU.
* **Мониторинг**: Логирование функции потерь в реальном времени и автоматическое сохранение результатов генерации каждую эпоху для отслеживания прогресса.

---

## Архитектура системы

### Генератор ([generator.py](https://www.google.com/search?q=./generator.py))

* **Вход**: 100-мерный латентный вектор.
* **Слои**: 4 последовательных блока ConvTranspose2d.
* **Активация**: `ReLU` во внутренних слоях и `Tanh` на выходе для формирования корректного цветового диапазона.
* **Выход**: RGB изображение размером .

### Дискриминатор ([discriminator.py](https://www.google.com/search?q=./discriminator.py))

* **Задача**: Определение подлинности изображения.
* **Слои**: Сверточные слои с использованием LeakyReLU для предотвращения проблемы "умирающих" нейронов.
* **Выход**: Сигмоидная функция для оценки вероятности того, что кадр является реальным.

---

## Результаты обучения

Процесс эволюции генератора от случайного шума до приближенных образов:

### Лес

| Эпоха 10 | Эпоха 2000 |
| --- | --- |
| <img width="300" height="150" alt="two_generated_epoch_10" src="https://github.com/user-attachments/assets/77df49f8-590e-463b-a297-7402c0c76194" /> | <img width="300" height="150" alt="two_generated_epoch_2000" src="https://github.com/user-attachments/assets/27ac7e2e-fe92-4e37-a6cd-1cc4f59c78be" /> |

### Горы

| Эпоха 10 | Эпоха 500 |
| --- | --- |
| <img width="300" height="150" alt="two_generated_epoch_10" src="https://github.com/user-attachments/assets/6aa6bc5c-bce0-453f-a2dc-ac5aea42e3bb" /> | <img width= "300" height="150" alt="two_generated_epoch_500" src="https://github.com/user-attachments/assets/b6720eeb-7eed-4e58-9d6d-4f04baeeb82d" /> |

### Логотипы

| Эпоха 10 | Эпоха 2000 |
| --- | --- |
| <img width="300" height="150" alt="two_generated_epoch_10" src="https://github.com/user-attachments/assets/09af6f84-7ea5-4116-b71b-b907ab92fa9d" /> | <img width="300" height="150" alt="two_generated_epoch_2000" src="https://github.com/user-attachments/assets/48c992d1-5cfb-4ee8-9c6f-2a96ea252263" /> |

### Здания

| Эпоха 10 | Эпоха 500 |
| --- | --- |
| <img width="300" height="150" alt="two_generated_epoch_10" src="https://github.com/user-attachments/assets/c535b636-f22c-4ce8-8e46-89e800186ed0" /> | <img width="300" height="150" alt="two_generated_epoch_500" src="https://github.com/user-attachments/assets/cdd6af73-9f71-4dfc-bfe9-f4541ed9a408" /> |

---

## Cтек
* PyTorch
* Torchvision
* CUDA
* NumPy
