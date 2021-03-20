import collections
import math
import random as rand
import string
import re

import nltk
from nltk.book import *
from nltk.corpus import stopwords
from nltk import ngrams
import random
import time
import hashlib
import itertools
from functools import reduce
from typing import List, T, Dict

import numpy as np
import statistics as stc


#######################################################################
def task1():
    """
    Ask the user for a number. Check if the input is an integer. Depending on whether
    the number is odd or even, print out an appropriate message to the user
    """
    try:
        val = float(input('Enter a integer number: '))
        if val != int(val):
            print("I sad integer number...")
            return
        print('The number', val, 'is: ', 'even' if val % 2 == 0 else 'odd')
    except ValueError:
        print('Invalid number')


def task2():
    """
    Generate 20 integers in the range 10 − 99 at random and return their mean value
    and maximal value
    """
    ran_numb = [rand.randint(10, 99) for _ in range(20)]
    ran_numb.sort()
    return sum(ran_numb) / len(ran_numb), ran_numb[19]


def task3(vector_a: List[float], vector_b: List[float]):
    """
    Write a function that computes cosine of the angle between two d-dimensional vectors
    """
    assert len(vector_a) == len(vector_b), "Vectors have different dimensions"

    scalar_product: float = 0.0
    for i in range(len(vector_a)):
        scalar_product += (vector_a[i] * vector_b[i])

    vector_a_length = math.sqrt(reduce(lambda a, b: a + b * b, vector_a, 0))
    vector_b_length = math.sqrt(reduce(lambda a, b: a + b * b, vector_b, 0))

    return scalar_product / (vector_a_length * vector_b_length)


def task4(integers: List[int], divider: int):
    """
    Write a function that for a given set of integers {a, . . . , b} and an integer c lists all
    integers greater than a and smaller than b that are divisible by c.
    """
    assert len(integers) > 0, "Give sth more"
    assert divider != 0, "divider can not be 0"

    a = integers[0];
    b = integers[len(integers) - 1]
    integers.sort()
    print('Greater than a:', a, list(filter(lambda arg: arg > a, integers[integers.index(a):])))
    print('Smaller than b:', b, list(filter(lambda arg: arg < b, integers[:integers.index(b)])))
    print('Divisible by c:', divider, list(filter(lambda arg: arg % divider == 0, integers)))


def task5(list_x: List[T], list_y: List[T]):
    """
    Write a program that for two lists x and y (possibly of different sizes) returns a list
    of elements that are common between the lists
    """
    return list(set(list_x).intersection(set(list_y)))


def task6(word: str):
    """
    Write a function that removes all a letters form a given string x. For example, for
    x =abracadabra we expect x =brcdbr
    """
    return word.replace('a', '')


def task7():
    """
    Write a program that accepts a sentence and calculates the number of letters and
    digits. Hint: you can use functions isalpha() and isdigit().
    """
    input_word = input('Enter sth: ')
    letters = 0
    digit = 0
    for i in range(len(input_word)):
        if input_word[i].isdigit():
            digit += 1
        elif input_word[i].isalpha():
            letters += 1

    print(f'Numbers {digit} letters: {letters}')


def task8():
    """
    Write a program that lists all subsets of the set {a, b, c, d}, 2^n always
    """
    ex_set = ['a', 'b', 'c', 'd']
    print(reduce(lambda result, x: result + [subset + [x] for subset in result], ex_set, [[]]))


def task9():
    """
    Write a program that returns the most frequent letter in a string
    """
    input_str = input('Enter sth: ')
    histogram = collections.defaultdict(int)

    for c in input_str:
        histogram[c] += 1
    sorted_dict = {k: v for k, v in sorted(histogram.items(), key=lambda item: item[1], reverse=True)}
    most_common = -1
    for sign in sorted_dict.items():
        if sign[1] >= most_common:
            print(sign[0], end=',')
            most_common = sign[1]


def task10(to_convert: int):
    """
    Convert a number represented as a sequence of decimal digits to the binary representation
    """
    if to_convert == 0:
        print(0)
        return
    size = int(math.ceil(math.floor(math.log(math.fabs(to_convert)) / math.log(2))))
    if to_convert < 0:
        size = max(size, 32)  # lets say that we consider integers and bigger representation

    res = []
    for i in range(size, -1, -1):
        res.insert(0, str(to_convert & 1))
        to_convert >>= 1

    print("".join(res))


def task11(arg: List[int]):
    """
    From a given list of integers remove all negative numbers
    """
    return list(filter(lambda x: x >= 0, arg))


def task12(arg: List[str]):
    """
    From a list of strings remove all strings longer than 5
    """
    return list(filter(lambda x: len(x) <= 5, arg))


def task13(arg: List[str]):
    """
    Construct a function that returns a string of maximal length from a set of strings.
    Note that the the maximal string does not have to be unique
    """
    length = max(len(i) for i in arg)
    print("Maximal length string: ", [i for i in arg if len(i) == length])


def task14(list_a: List, list_b: List) -> List:
    """
    Construct a function that constructs a list that keeps alternately consecutive elements from two argument lists of
    equal length. For example, for a=[a1,a2,a3] and a=[b1,b2,b3] the correct output is [a1,b1,a2,b2,a3,b3] .
    """
    assert len(list_a) != list_b, 'list_a must the same length what list_b'

    result = []
    for i in range(len(list_a)):
        result.append(list_a[i])
        result.append(list_b[i])
    return result


def task15(arg: List):
    """
    A list contains both integers and strings. Construct a function that sorts given list
    in such a way that at the beginning of the resulting list one gets strings in alphabetical
    order followed by integers in an increasing order.
    """
    return sorted(filter(lambda x: str(x).isdigit(), arg)) + (sorted(filter(lambda x: str(x).isalpha(), arg)))


def task16():
    """
    Construct a dictionary that assigns each user (string) his address (string). Sort
    and list this dictionary by the addresses. Use lambda function.
    """
    users = ['Bob', 'Thomas', 'John', 'Zbigniew', 'Zdzisiu']
    addresses = ['New York', 'London', 'Tokyo', 'Warszawa', 'Wroclaw']
    user_on_address = dict(zip(users, addresses))
    print({k: v for k, v in sorted(user_on_address.items(), key=lambda item: item[1])})


# def task17():
#     """
#     Import all books from nktl.book. Check how many times the word knight appe-
#     ars in text6 (Monty Python) and text7
#     """
#     # Before
#     # import nltk
#     # nltk.download('gutenberg') in terminal
#     # nltk.download('genesis') in terminal
#     # nltk.download('nps_chat')
#     # nltk.download('webtext')
#     # nltk.download('treebank')
#
#     print(f'Word knight occurs {text6.count("knight")} times in text6 and {text7.count("knight")} in text7 ')
#
#
# def task18():
#     """
#     Import all books from nktl.book. Construct a set of words that appear in in
#     text6 (Monty Python) but do not appear in text7 (Wall Street Journal).
#     """
#     return set(text6.tokens).intersection(set(text7.tokens))
#
#
# def task19():
#     """
#     Import all books from nktl.book. Construct a set of words that appear in in all
#     texts text1 - text9.
#     """
#
#     text_1 = set(text1)
#     text_2 = set(text2)
#     text_3 = set(text3)
#     text_4 = set(text4)
#     text_5 = set(text5)
#     text_6 = set(text6)
#     text_7 = set(text7)
#     text_8 = set(text8)
#     text_9 = set(text9)
#
#     duplicates = []
#     # append the same words if they are similar
#     for w in text_1:
#         if w in text_2 and text_3 and text_4 and text_5 and text_6 and text_7 and text_8 and text_9:
#             duplicates.append(w)
#
#     from nltk.corpus import stopwords
#     stop_words = set(stopwords.words('english'))
#
#     for w in duplicates:
#         if w in stop_words:
#             duplicates.remove(w)
#
#     return duplicates
#
#
# def task20():
#     """
#     Find the longest sentence in text2
#     """
#     sign = str.maketrans('', '', '--')
#     translated_text = [x.translate(sign) for x in text2]
#     text = ""
#     for x in translated_text:
#         if x in string.punctuation or x == "s":
#             text = text + x
#         else:
#             text = text + " " + x
#
#     print(max(nltk.sent_tokenize(text), key=lambda txt: len(nltk.word_tokenize(txt))))


def task21(bit_string: str):
    """
    For a given bitstring b list all bitstrings b’, such that the Hamming distance between
    b and b’ is equal 1.
    """
    assert len(re.sub('1|0', '', bit_string)) == 0, "Enter correct string"
    for i in range(len(bit_string)):
        replacement = '0' if bit_string[i] == '1' else '1'
        print(bit_string[:i] + replacement + bit_string[i + 1:])


def task22(first_set: List, second_set: List) -> None:
    """
    Construct a function that returns a Jaccard similarity for two sets. Beware that this
    function needs to check if at least one of the sets is nonempty.
    J(A,B) = |A ∩ B| / |A u B|
    """
    s1 = set(first_set)
    s2 = set(second_set)
    intersection = s1.intersection(s2)
    union = s2.union(s1)
    if len(union) == 0:
        print("Can not calculate Jaccard similarity due tu |A u B| = 0")
    else:
        print("Jaccard similarity: ", (len(intersection) / float(len(union))))


def task23(str_one: str, str_two: str) -> None:
    """
    Construct a function that computes Jaccard similarity for two strings treated as bags
    of words.
    """
    regex = r'\s|[!"#$%&\'()*+,.\/:;<=>?@\^_`{|}~-]+'
    s1 = set(re.split(regex, str_one))
    s2 = set(re.split(regex, str_two))
    intersection = s1.intersection(s2)
    union = s2.union(s1)
    if len(union) == 0:
        print("Can not calculate Jaccard similarity due tu |A u B| = 0")
    else:
        print("Jaccard similarity: ", (len(intersection) / float(len(union))))


# def task24() -> None:
#     """
#     (use NLTK) List all words in text1 with edit distance from the word dog smaller
#     than 4. Hint: you can safely reject all long words without computations (why?).
#     """
#     print(list(filter(lambda x: len(x) <= 3 or nltk.edit_distance(x, "dog") < 4, set(text1))))


# def task25() -> None:
#     """
#     (use NLTK) Let text1 - text9 be bags of words. Compute similarity between all
#     pairs of texts.
#     """
#     words_bags = list(map(lambda x: set(x), [text1, text2, text3, text4, text5, text6, text7, text8, text9]))
#     for a in words_bags:
#         for b in words_bags:
#             intersection = a.intersection(b)
#             union = a.union(b)
#             if len(union) == 0:
#                 print("Can not calculate Jaccard similarity due tu |A u B| = 0")
#             else:
#                 print("Jaccard similarity: ", (len(intersection) / float(len(union))))
#
#
# def task26() -> None:
#     """
#     (use NLTK) Let us consider a metric space (S, d), where S is the set of words from
#     text1 and d is the Hamming distance. Find diameter of (S, d).
#     Notes: The diameter of a set in a metric space is the supremum of distances between its points
#     """
#     punctions = str.maketrans('', '', string.punctuation)
#     stop_words = set(stopwords.words('english'))
#     cleared_text = [word.translate(punctions) for word in text1 if word not in stop_words]
#     none_empty_words = [x for x in set(cleared_text) if x != '']
#     max_distance = 0
#     hamming_distance = lambda x, y: sum(letter_left != letter_right for letter_left, letter_right in zip(x, y))
#     for x_pair in none_empty_words:
#         for y_par in none_empty_words:
#             if x_pair != y_par:
#                 distance = hamming_distance(x_pair, y_par) if len(x_pair) == len(y_par) else 0
#                 if distance > max_distance:
#                     max_distance = distance
#
#     print("Diameter for text1 is: " + str(max_distance))


# def task27() -> None:
#     """
#     (use NLTK) Construct a dictionary that assigns each pair of consecutive words
#     in text1 the Jaccard similarity between them.
#     """
#     punctions = str.maketrans('', '', string.punctuation)
#     stop_words = set(stopwords.words('english'))
#     cleared_text = [word.translate(punctions) for word in text1 if word not in stop_words]
#
#     def jaccard_similarity(x: set, y: set):
#         intersection = x.intersection(y)
#         union = x.union(y)
#         return 0 if len(union) == 0 else (len(intersection) / float(len(union)))
#
#     word_pair_on_jaccard_similarity = {}
#
#     for i in range(len(cleared_text) - 1):
#         key = cleared_text[i] + ':' + cleared_text[i + 1]
#         word_pair_on_jaccard_similarity[key] = jaccard_similarity(set(cleared_text[i]), set(cleared_text[i + 1]))
#     print(word_pair_on_jaccard_similarity)


# def task28() -> None:
#     """
#     (use NLTK). For two words v and w, let relative edit distance be the Levensthein
#     distance between v and w divided by the sum of lengths v and w. Find two different
#     words in text2 with minimal relative edit distance.
#     Notes: Levenshtein distance between two words is the minimum number of single-character edits
#     (insertions, deletions or substitutions) required to change one word into the other.
#     """
#     punctions = str.maketrans('', '', string.punctuation)
#     stop_words = set(stopwords.words('english'))
#     text_2_words = set(text2)
#     cleared_text = [word.translate(punctions) for word in text_2_words if word not in stop_words]
#     cleared_text.sort(key=len, reverse=True)
#     cleared_text = cleared_text[:1000]
#
#     edit_distances = [(10_000, ""), (10_000, "")]
#     for i in range(len(cleared_text)):
#         for j in range(i + 1, len(cleared_text)):
#             distance = nltk.edit_distance(cleared_text[i], cleared_text[j]) / float(
#                 len(cleared_text[i]) + len(cleared_text[j]))
#             if distance < edit_distances[0][0]:
#                 edit_distances[0] = (distance, cleared_text[i] + ":" + cleared_text[j])
#             elif distance < edit_distances[1][0]:
#                 edit_distances[1] = (distance, cleared_text[i] + ":" + cleared_text[j])
#     print(edit_distances)


def task29(bit_string: str, r: int) -> None:
    """
    For a given bitstring b and a natural number r list all bitstrings b’, such that the
    Hamming distance between b and b’ is equal n.
    Notes: Should be 'is equal r' instead 'n' ??? let's assume that 'n'
    """
    result = []
    cartesian_product = ["".join(map(str, x)) for x in itertools.product([0, 1], repeat=len(bit_string))]

    for bit_str in cartesian_product:
        distance = sum(w1 != w2 for w1, w2 in zip(bit_str, bit_string))
        if distance == r:
            result.append(bit_str)

    print(result)


def task30(some_string: str, k: int) -> None:
    """
    Construct a function that for a given string and a natural number k returns a set
    of all its k-shingles.
    """
    assert k > 0
    tokens = some_string.split()
    print([tokens[i:i + k] for i in range(len(tokens) - k + 1)])


def task31() -> None:
    """
    Generate a set S of n random bitstrings of length 100. Find min x,y∈S sha-1(x||y), where
    x||y denotes concatenation of bitstrings x and y. Estimate, what is the maximal n for
    this task that can be handled by your computer?
    Notes : x||y and also  y||x ??, b) I can leave my computer for 100 years to keep it count... ?
    """

    for test in range(1, 10):
        start_time = time.time()
        bit_strings = ["{0:b}".format(random.getrandbits(100)).zfill(100) for _ in range(test * 1000)]
        y = hashlib.sha1(bytearray("".join(["1" for _ in range(200)]), "utf8")).digest()
        for i in range(len(bit_strings)):
            for j in range(i + 1, len(bit_strings)):
                x = hashlib.sha1(bytearray(bit_strings[i] + bit_strings[j], "utf8")).digest()
                y = min(x, y)
        print("--- %s seconds --- %s" % (time.time() - start_time, test))
        # print(f'Min: {y}')


def task32() -> Dict:
    """
    (use NLTK). Let S1 , S2 , S3 be the texts of all words shorter than 8 letters from text1,
    text2, text3, respectively. Compute signatures for S1 , S2 , S3 represented by 100
    minhashes and then estimate Jaccard similarity between each pair of S1 , S2 , S3 .
    """
    texts = ['text1', 'text2', 'text3']
    txt_1 = [t for t in text1 if len(t) > 8]
    txt_2 = [t for t in text2 if len(t) > 8]
    txt_3 = [t for t in text3 if len(t) > 8]
    texts_set = [set(txt_1), set(txt_2), set(txt_3)]

    def min_hash_signature(text_set: set) -> List[float]:
        return sorted(hash(word) for word in text_set)[:100]

    signatures = [min_hash_signature(s) for s in texts_set]
    jaccards = []
    for i in range(len(texts_set)):
        for j in range(i + 1, len(texts_set)):
            set_sign1 = set(signatures[i])
            set_sign2 = set(signatures[j])
            intersection = set_sign1.intersection(set_sign2)
            union = set_sign1.union(set_sign2)
            jaccards.append(len(intersection) / len(union))
    return {
        txt1 + ':' + txt2: jaccard
        for ((txt1, txt2), jaccard) in zip(itertools.combinations(texts, 2), jaccards)
    }


def task33() -> None:
    """
    Compare the results from the previous exercise with the exact Jaccard similarity of
    sets S 1 , S 2 , S 3 . What if random permutation of the characteristic matrix rows were
    replaced with a random mapping?
    """


if __name__ == '__main__':
    # task1()
    # print(task2())
    # print(task3([4, 3], [2, 5]))
    # task4([2, 8, 12, 45, 7, 3, 45, 23, 4, 23, 11, 1, -1, 0, 5], 3)
    # print(task5([1, 2, 3, 6, 7], [-1, -2, 3, 10, 5]))
    # print(task5([1, 2, 3, 6, 7], ['a', 'b', 'c', 'd', 1]))
    # print(task6("abracadabra"))
    # task7()
    # task8()
    # task9()
    # task10(-5)
    # print(task11([-1, -2, -3, -4, -5, -6, -7, -8, -9, -0, 1, 2, 3, 4, 5]))
    # print(task12(['aaaaaaaa', 'aaaa', '', 'aaaa', 'aaaaaa', ]))
    # task13(['cccccccc', 'cccc', '', 'cccc', 'cccccc', 'dddddddd', 'dddd', '', 'dddd', 'dddddd', 'eeeeeeee', 'eeee', '',
    #         'eeee', 'eeeeee', 'ffffffff', 'ffff', '', 'ffff', 'ffffff'])
    # print(task14([1, 2, 3], ['a', 'b', 'c']))
    # print(task15([1, 2, 3, 'a', 'b', 'c', 'z', 'x', 'u', -1, -2, 0]))
    # print(task16())
    # task17()
    # print(task18())
    # task19()
    # task20()
    # task21("1111000")
    # task22([1, 2, 3, 45, 'a', 1.333], [1, 2, 3, 45, 'b', 1.2222])

    #     task23(""""    Construct a function that computes Jaccard similarity for
    #     two strings treated as bags of words.""",
    #            """Construct a function that returns a Jaccard similarity for two sets. Beware that this
    # function needs to check if at least one of the sets is nonempty.""")
    # task24()
    # task25()
    # task26()
    # task27()
    # task28()
    # task29("110", 2)
    # task30("Construct a function that for a given string and a natural number k", 3)
    # task31()
    print(task32())
    print()
