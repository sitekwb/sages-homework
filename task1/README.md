# Parser kostek domina

Aby skonfigurować środowisko wirtualne Python, należy uruchomić komendy:  
`conda create -n sages-homework python=3.9`  
`conda activate sages-homework`  
`pip install -r requirements.txt`  

Aby uruchomić program, należy w katalogu roboczym `task1` uruchomić komendę:  
`python src/main.py [-h] -c CODE -i ITER [-r | --reverse | --no-reverse]`  
`CODE` oznacza kostki domina, `ITER` oznacza liczbę iteracji, a `-r` określa, czy program działa w odwrotnym kierunku.  

Aby uruchomić testy, należy wpisać komendę:  
`python -m unittest test.DominoParserTest`
