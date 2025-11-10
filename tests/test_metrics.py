from suffixranker.utils import mapk

def test_mapk_basic():
    y_true = [0, 1, 2]
    y_pred = [[0,2,1], [0,1,2], [1,2,0]]
    score = mapk(y_true, y_pred, k=3)
    # APs: 1.0, 0.5, 0.5 => mean 0.666...
    assert 0.66 < score < 0.67
