# Bias


Bias dla pojedynczego neuronu jest zazwyczaj inicjalizowany na wartość losową, ale istnieją różne strategie inicjalizacji, a nie zawsze jest to wartość losowa. Poprawny wybór strategii inicjalizacji może wpłynąć na efektywność uczenia się sieci neuronowej.

Najczęściej stosowane metody inicjalizacji biasa obejmują:

1. **Inicjalizacja losowa:** Bias jest inicjalizowany losowo, na przykład z rozkładem jednorodnym lub normalnym. Warto zauważyć, że ważne jest, aby wagi i biasy były inicjalizowane w sposób zapewniający zrównoważone uczenie się sieci.
2. **Inicjalizacja zerowa:** Bias jest inicjalizowany wartością zerową. Choć to jedna z prostszych strategii, czasami może działać w praktyce, szczególnie w przypadku pewnych architektur sieci.
3. **Inicjalizacja xavier/glorot:** To bardziej zaawansowana metoda inicjalizacji, która próbuje utrzymać wariancję sygnału wewnątrz sieci. Jest szczególnie popularna w sieciach z funkcją aktywacji sigmoidalną lub tangens hiperboliczny.
4. **Inicjalizacja He:** Podobna do inicjalizacji xavier, ale często stosowana w sieciach z funkcją aktywacji ReLU.





Bias (ang. "obciążenie" lub "strzałka") to dodatkowy parametr w sieciach neuronowych, który jest używany do przesunięcia aktywacji neuronu. W kontekście sieci neuronowych, bias jest dodawany do sumy ważonych wejść neuronu przed zastosowaniem funkcji aktywacji. Bias pozwala na przesunięcie funkcji aktywacji w górę lub w dół, co jest istotne dla efektywności i elastyczności modelu.

Konkretniej, dla jednego neuronu, sumę ważonych wejść (oznaczmy ją jako z**z**) można wyrazić jako:


gdzie:

* wi**w**i**** to wagi dla poszczególnych wejść xi**x**i****,
* b**b** to bias.
