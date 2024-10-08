import sys,random
from ezr import the, DATA, csv
import copy

def testing_tools():
    ## Testing whether chebyshevs().rows[0] returns the top item of the sorted list
    def test_cheby():
        print("----------------Testing chebyshevs----------------")
        d = DATA().adds(csv("data/optimize/misc/auto93.csv"))
        dumb = d.clone(random.choices(d.rows, k=20)).chebyshevs().rows
        print(f'Sorted list[:5]     :\n{dumb[:5]}\n\n')
        print(f'Cheby distances of first five items: {[ round(d.chebyshev(du),3) for du in dumb[:5] ]}\n')
        print(f'Top Item and its cheby dist in the list: {dumb[0], round(d.chebyshev(dumb[0]),3)}\n')
        [print(f'chebyshevs works fine') if d.chebyshev(dumb[0]) == min([d.chebyshev(du) for du in dumb]) else print(f'chebyshevs has a bug')]
        print("----------------chebyshevs test ends----------------\n\n")

    ## Testing whether the shuffle method really works
    def test_shuffle():
        print("----------------Testing Shuffle----------------")
        d = DATA().adds(csv("data/optimize/misc/auto93.csv"))
        print(f'The original dataset [0:5] \t\tAfter Shuffling [0:5]')
        [print(f'{i}\t{j}') for i,j in zip(d.rows[:5],d.shuffle().rows[:5])]
        [print('shuffle works properly') if copy.deepcopy(d.rows)!=d.shuffle().rows else print('shuffle is not working')]
        print("----------------Shuffle test ends----------------\n\n")

    ## Testing whether smart and dumb lists has the right lenght
    def test_lenght():
        print("----------------Testing Lenghts----------------")
        d = DATA().adds(csv("data/optimize/misc/auto93.csv"))
        the.Last = N = 20
        dumb = [d.clone(random.choices(d.rows, k=N)).chebyshevs().rows]
        smart = [d.shuffle().activeLearning()]
        print(f'N is {N}, lenghts of dumb and smart are {len(dumb[0])}, {len(smart[0])}')
        [print(f'Lenghts are right') if N==len(dumb[0])==len(smart[0]) else print(f'Lenghts are not right')]
        print("----------------Lenghts test ends----------------\n\n")

    ## Testing whether experiments are done 20 times
    def test_repeats():
        print("----------------Testing Repeats----------------")
        d = DATA().adds(csv("data/optimize/misc/auto93.csv"))
        repeats = 20
        dumb = [d.clone(random.choices(d.rows, k=30)).chebyshevs().rows for _ in range(repeats)]
        smart = [d.shuffle().activeLearning() for _ in range(repeats)]
        d_unique = []
        [d_unique.append(du) for du in dumb if du not in d_unique]
        s_unique = []
        [s_unique.append(s) for s in smart if s not in s_unique]
        print(f"Number of similar treatment settings among {repeats} repeats: {len(dumb)-len(d_unique)} for dumb and {len(smart)-len(s_unique)} for smart")
        [print(f'Repeats are valid') if (len(d_unique)==len(dumb) and len(s_unique)==len(smart)) else print(f'Repeats are not valid')]
        print("----------------Lenghts Repeats ends----------------")


    test_shuffle()
    test_lenght()
    test_cheby()
    test_repeats()

testing_tools()