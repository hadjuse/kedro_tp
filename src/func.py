# Ecrire dans un fichier indépendant du projet nommé pytest_1.py
def argmax(liste):
    if len(liste) == 0:
        return None

    idx_max = 0
    value_max = liste[0]
    for i, x in enumerate(liste):
        if x > value_max:
            value_max = x
            idx_max = i
    return idx_max


def test_argmax():
    assert argmax([5, 8, 2, 9, 6, 3]) == 3
    assert argmax([7]) == 0
    assert argmax([]) is None
