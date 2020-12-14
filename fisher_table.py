import argparse
import collections
import itertools
import random
import time


def check_table(table):
    last_p_set = None
    for p in table:
        p_set = set(p)
        assert len(p_set) == len(p)
        if last_p_set is None:
            last_p_set = p_set
        else:
            assert p_set == last_p_set


def print_table_stats(table):
    sums = collections.defaultdict(int)
    for p in table:
        for i, v in enumerate(p, start=1):
            sums[v] += i

    min_sum = min(sums.values())
    max_sum = max(sums.values())

    m = len(table)
    n = len(sums)
    balance = (n + 1) / 2.
    print('Ожидаемое среднее место {:g}.'.format(balance), end=' ')
    if min_sum == max_sum:
        print('Расписание сбалансировано.')
    else:
        print('Диапазон средних мест в таблице от {:.4g} до {:.4g}.'.format(
            min_sum / m,
            max_sum / m
        ))


def print_table_1(table):
    n = max(map(len, table))

    title = range(1, n + 1)
    place_width = max(map(len, map(str, itertools.chain(*table, title))))

    def print_list(lst):
        print(' '.join(map(lambda v: str(v).rjust(place_width), lst)))

    print('-' * (n * place_width + n - 1))
    print_list(title)
    print('-' * (n * place_width + n - 1))
    for p in table:
        print_list(p)
    print('-' * (n * place_width + n - 1))


def print_table_2(table):
    place_width = max(map(len, map(str, itertools.chain(*table))))
    table2 = collections.defaultdict(list)
    for p in table:
        for i, j in enumerate(p, 1):
            table2[j].append(i)
    table2 = list(table2.items())
    table2.sort()
    for k, lst in table2:
        print('{}: {} : {:.4g}'.format(
            str(k).rjust(place_width),
            ' '.join(map(lambda v: str(v).rjust(place_width), lst)),
            sum(lst) / len(lst)
        ))


def search_permutations(values, callback):
    lst = list(values)
    n = len(lst)

    def swap(i, j):
        lst[i], lst[j] = lst[j], lst[i]

    def search(k):
        if k < n:
            for i in range(k, n):
                swap(k, i)
                callback(k + 1, k, lst[k], lambda: search(k + 1))
                swap(k, i)

    search(0)


class TimeIsOut(Exception):
    pass


class TableIsReady(Exception):
    def __init__(self, table):
        super(TableIsReady, self).__init__('')
        self.table = table


def restricted_table(n, limit, rnd):
    assert n > 0

    first = list(range(n))
    second = [None] * n
    third = [None] * n

    s = 3 * (n - 1) // 2
    if n % 2 == 0:

        def kks(k1k2):
            k3 = s - k1k2
            return k3, k3 + 1

    else:

        def kks(k1k2):
            return s - k1k2,

    def callback(k, k2, k1, search_deeper):
        nonlocal count
        if count >= limit:
            raise TimeIsOut
        count += 1

        assert second[k2] is None
        second[k2] = k1
        for k3 in kks(k2 + k1):
            if 0 <= k3 < n and third[k3] is None:
                third[k3] = k1
                if k == n:
                    raise TableIsReady((first, second, third))
                search_deeper()
                assert third[k3] is not None
                third[k3] = None
        assert second[k2] == k1
        second[k2] = None

    values = list(range(n))
    rnd.shuffle(values)

    count = 0
    try:
        search_permutations(values, callback)
    except TimeIsOut:
        return None
    except TableIsReady as e:
        return e.table

    assert False
    return None


def small_table(n, rnd):
    for p in itertools.count():
        limit = 2 ** p
        t = restricted_table(n, limit, rnd)
        if t is not None:
            return t


def merge_permutations(p0, p1):
    n0 = len(p0)
    n1 = len(p1)

    def interleave(n0, n1):
        n = min(n0, n1)
        for _ in range(n):
            yield from (0, 1)
        yield from itertools.repeat(0, n0 - n)
        yield from itertools.repeat(1, n1 - n)

    m0 = n0 // 2
    m1 = n1 // 2

    bits = itertools.chain(
        interleave(m0, m1),
        reversed(tuple(interleave(n0 - m0, n1 - m1)))
    )

    index = [[], []]
    for i, b in enumerate(bits):
        index[b].append(i)

    p = [None] * (n0 + n1)
    for ii, pi in zip(index, (p0, p1)):
        for i, v in enumerate(pi):
            p[ii[i]] = ii[v]
    return p


def plain_table(n, m, rnd):
    if n <= m:
        return small_table(n, rnd)

    n2 = n // 2
    n1 = n - n2
    t1 = plain_table(n1, m, rnd)
    t2 = plain_table(n2, m, rnd)

    return tuple(merge_permutations(p1, p2) for p1, p2 in zip(t1, t2))


def permuted_table(n, m, rnd):
    ids = list(range(1, n + 1))
    rnd.shuffle(ids)
    return tuple(tuple(ids[i] for i in p) for p in plain_table(n, m, rnd))


def main():
    parser = argparse.ArgumentParser(
        description='Турнирная таблица для соревнований рыбаков в три тура.'
    )
    parser.add_argument('n', metavar='N', type=int, help='число участников')
    parser.add_argument(
        '--seed',
        metavar='N',
        type=int,
        default=None,
        help='начальное состояние генератора случайных чисел'
    )
    parser.add_argument(
        '--small',
        metavar='N',
        type=int,
        default=25,
        help='меньше этого размера ищется наилучшее решение (25 по умолчанию)'
    )

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.Random(time.time_ns()).randrange(10 ** 6)
        print(
            "Используйте '--seed {}' чтобы воспроизвести результаты.\n".format(
                args.seed
            )
        )

    table = permuted_table(args.n, args.small, random.Random(args.seed))
    check_table(table)
    print_table_1(table)
    print()
    print_table_stats(table)
    print()
    print_table_2(table)


if __name__ == "__main__":
    main()
