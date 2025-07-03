#  BersbotsAILib

**BersbotsAILib** — это простая, самостоятельная (header-only), утилитная библиотека на C++17+ для создания и обучения нейросетей **без внешних зависимостей**.

---

##  Возможности

- Построение нейросети произвольной архитектуры: `NeuralNetwork({входы, скрытые..., выходы})`
- Обучение методом **обратного распространения ошибки (backpropagation)**
- Предсказания: `predict(...)`
- Сброс весов: `resetTrain()`
- Сохранение и загрузка весов: `saveWeights(...)` и `loadWeights(...)`
- Использование функции активации **sigmoid**
- Работа с пользовательскими датасетами

---

##  Пример использования

```cpp
#include "BersbotsAILib.hpp"
#include <iostream>

int main() {
    NeuralNetwork net({2, 3, 1}); // сеть: 2 входа → 3 скрытых → 1 выход

    std::vector<std::pair<std::vector<float>, std::vector<float>>> dataset = {
        { {0, 0}, {0} },
        { {0, 1}, {1} },
        { {1, 0}, {1} },
        { {1, 1}, {0} }
    };

    net.trainDataset(dataset, 10000, true, 0.1f);

    for (const auto& [input, target] : dataset) {
        auto output = net.predict(input);
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << output[0]
                  << " (ожидалось " << target[0] << ")\n";
    }

    net.saveWeights("weights.txt");
    net.resetTrain();
    net.loadWeights("weights.txt");

    return 0;
}
```
Классы и методы
🔹 class NeuralNetwork

    NeuralNetwork(std::vector<int> layers) — создать сеть

    predict(input) — получить предсказание

    train(input, target, learningRate) — обучить один пример

    trainDataset(dataset, epochs, showProgress, learningRate) — обучить весь датасет

    saveWeights(filename) / loadWeights(filename) — сохранить или загрузить веса

    resetTrain() — переинициализировать сеть

🔹 class Layer

    Представляет один слой сети

    Содержит веса, смещения, входы и выходы слоя

Требования

    C++17 или новее

    Только стандартная библиотека, никаких зависимостей

Установка

Просто скопируй файл BersbotsAILib.hpp в свой проект и подключи:

#include "BersbotsAILib.hpp"

Лицензия

MIT — используй свободно в учебных, исследовательских и личных целях.

Создан с нуля с любовью к C++ и нейросетям
Автор: [Bersbot](https://github.com/Bersbot) 

Но я понимаю, что эта библиотека не будет особо полезна для программистов, так как существуют и более продвинутые аналоги. В крайнем случае, можно сгенерировать такую же с помощью ChatGPT или DeepSeek.
Однако я делал её в обучающих целях. Также я хочу собрать команду программистов для работы над своими проектами и постараюсь сам писать большинство библиотек, которые буду использовать.
Я только в начале своего пути — мне пока что 13 лет.
