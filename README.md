# fisher_table

Программа для составления турнирных таблиц. Соревнование состоит из трёх туров. В каждом из туров участники расставляются в случайном порядке. Среднее место всех участников в таблице примерно одинаково, так чтобы участники не могли получить преимущество попадая в начало (или конец) списка во всех турах.

    $ python fisher_table.py 10
    Используйте '--seed 561571' чтобы воспроизвести результаты.

    Ожидаемое среднее место 5.5. Диапазон средних мест в таблице от 5.333 до 5.667.

    -----------------------------
     1  2  3  4  5  6  7  8  9 10
    -----------------------------
     9  5  3  4  6  1 10  2  8  7
    10  8  3  6  7  5  2  1  4  9
     7  2  1  4  8  9  6 10  5  3
    -----------------------------

Первая строка напоминает что используется генератор случайных чисел. Зафиксировать результат можно с помощью `--seed`.
Вторая строка выводит ожидаемое среднее место в таблице. Реальные средние места могут отличаться от ожидаемого. Для оценки качества результата выводится диапазон средних мест участников в этой конкретной таблице.

Последней напечатана сама таблица. В заголовке - номер места, в теле - номера участников в каждом из трёх туров.

В некоторых ситуациях удаётся построить полностью сбалансированную таблицу.

    $ python fisher_table.py 11
    Используйте '--seed 14330' чтобы воспроизвести результаты.

    Ожидаемое среднее место 6. Расписание сбалансировано.

    --------------------------------
     1  2  3  4  5  6  7  8  9 10 11
    --------------------------------
     1 11  4  3  6  9  7  8 10  5  2
     7  8  9  2 11  5  3 10  6  4  1
    10  5  2  6  4  1  3  8  9  7 11
    --------------------------------

