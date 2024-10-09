import random
from ezr import the, DATA, csv, NUM, SYM, row, generate_new_value

def upsampling(self:DATA, new:int) -> DATA:
    new_rows = []
    # [print(col.txt, end=",\t") for col in self.cols.all]
    # print()
    for _ in range(new):
        r1, r2, r3 = random.choices(self.rows, k=3)
        new_row = []
        for c in self.cols.all:
            random_number = random.random()
            if random_number < 0.9:
                new_val = c.generate_new_value(r1, r2, r3)
                new_row.append(new_val)
            else:
                new_row.append(r1[c.at])
        new_rows += [new_row]
    # new_d = self.clone(new_rows)
    # [print(nr) for nr in self.rows]
    # print('------------------')
    # [print(nr) for nr in new_rows] 
    # print('-----------------')

    # for dd, nd in zip(self.cols.all, new_d.cols.all):
        
    #     print(f"d  : div={dd.div()}, mid={dd.mid()} --- {dd.txt}")
    #     print(f"syn: div={nd.div()}, mid={nd.mid()}")
    #     print()
    # input()
    return self.clone(self.rows + new_rows)

if __name__ == '__main__':
    d = DATA().adds(csv("data/optimize/misc/auto93.csv"))
    random_rows = d.clone(random.choices(d.rows, k=the.Stop))

    upsampling(random_rows, 10)

# print(len(random_rows.rows))
# for r in random_rows.rows:
#     print(r)
