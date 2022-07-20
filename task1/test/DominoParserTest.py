import unittest

from DominoParser import DominoParser


class DominoParserTest(unittest.TestCase):
    def test_two_characters(self):
        parser = DominoParser('/|', 1, reverse=False)
        self.assertEqual(parser.parse(), '//')

    def test_two_characters_many_iterations(self):
        parser = DominoParser('/|', 5, reverse=False)
        self.assertEqual(parser.parse(), '//')

    def test_right_change(self):
        parser = DominoParser('||||\\\\\\', 3, reverse=False)
        self.assertEqual(parser.parse(), '|\\\\\\\\\\\\')

    def test_long(self):
        parser = DominoParser('||||\\\\\\///||||\\\\//\\\\/|||/', 3, reverse=False)
        self.assertEqual(parser.parse(), '|\\\\\\\\\\\\/////\\\\\\\\//\\\\/////')

    def test_reverse_really_short(self):
        parser = DominoParser('/|', 1, reverse=True)
        self.assertEqual(parser.parse(), '/|')

    def test_reverse_short(self):
        parser = DominoParser('////|', 5, reverse=True)
        self.assertEqual(parser.parse(), '/||||')

    def test_reverse_long(self):
        parser = DominoParser('////||||\\\\\\\\|||||////', 3, reverse=True)
        self.assertEqual(parser.parse(), '/||||||||||\\|||||/|||')

    def test_example(self):
        parser = DominoParser('||//||\\||/\\|', 1, reverse=False)
        self.assertEqual(parser.parse(), '||///\\\\||/\\|')

    def test_reverse_example(self):
        parser = DominoParser('||////\\\\\\|////|', 2, reverse=True)
        self.assertEqual(parser.parse(), '||//||||\\|//|||')


if __name__ == '__main__':
    unittest.main()
