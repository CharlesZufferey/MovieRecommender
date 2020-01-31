import random
import sqlalchemy

MOVIES = [  'Star Wars 1',
            'Star Wars 2',
            'Star Wars 3',
            'Star Wars 4',
            'Star Wars 5',
            'Star Wars 6',
            'Star Wars 7',
            'Star Wars 8',
            'Star Wars 9',
            'Shawshank Redemption',
            'Docker Unleashed',
            'Django Unchained',
            'Flask Unchained',
            'Some awesome movie'
]

def recommend(n):
    """Given a number, returns that many movies as a list"""
    result_list = random.sample(MOVIES, k=n)
    return result_list