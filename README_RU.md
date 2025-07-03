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
