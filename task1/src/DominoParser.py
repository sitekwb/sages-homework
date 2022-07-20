from typing import List, Tuple, Optional
from more_itertools import triplewise


class DominoParser:
    def __init__(self, input_string: str, iter_count: int, reverse: bool):
        self.input: List[str] = list(input_string)
        self.iter_count: int = iter_count
        self.reverse: bool = reverse
        self.groups: List[Tuple[int, int, str]] = []
        self.divide_into_groups()
        self.out = self.input

    def divide_into_groups(self):
        current_domino_group: Optional[str] = self.input[0] if len(self.input) > 0 else '|'
        start: int = 0
        for i, domino in enumerate(self.input):
            if domino != current_domino_group:
                self.groups.append((start, i, current_domino_group))
                start = i
                current_domino_group = domino
        self.groups.append((start, len(self.input), current_domino_group))

    def change(self, start: int, end: int, from_left: bool = False, from_right: bool = False):
        a: int = start
        b: int = end - 1
        j: int = 0
        while a <= b and j < self.iter_count:
            if from_left and from_right and a == b:
                break
            if from_left:
                self.out[a] = '/'
                a += 1
            if from_right:
                self.out[b] = '\\'
                b -= 1
            j += 1

    def reverse_change(self, start: int, end: int, from_right: bool = False):
        a: int = start
        b: int = end - 1
        j: int = 0
        while a < b and j < self.iter_count:
            if from_right:
                self.out[b] = '|'
                b -= 1
            else:
                self.out[a] = '|'
                a += 1
            j += 1

    def run_forward(self):
        if len(self.groups) > 1:
            start1, end1, d1 = self.groups[0]
            start2, end2, d2 = self.groups[1]
            l_start1, l_end1, l_d1 = self.groups[-2]
            l_start2, l_end2, l_d2 = self.groups[-1]
            if d1 == '|' and d2 == '\\':
                self.change(start1, end1, from_right=True)
            if l_d2 == '|' and l_d1 == '/':
                self.change(l_start2, l_end2, from_left=True)
        for g1, g2, g3 in triplewise(self.groups):
            start, end, domino = g2
            if domino != '|':
                continue
            _, _, previous_domino = g1
            _, _, next_domino = g3
            if previous_domino == '/' and next_domino == '\\':
                self.change(start, end, True, True)
            elif previous_domino == '/' and next_domino == '/':
                self.change(start, end, from_left=True)
            elif previous_domino == '\\' and next_domino == '\\':
                self.change(start, end, from_right=True)
        return ''.join(self.out)

    def run_backward(self):
        for start, end, domino in self.groups:
            if domino == '/':
                self.reverse_change(start, end, from_right=True)
            elif domino == '\\':
                self.reverse_change(start, end, from_right=False)
        return ''.join(self.out)


    def parse(self) -> str:
        return self.run_backward() if self.reverse else self.run_forward()
