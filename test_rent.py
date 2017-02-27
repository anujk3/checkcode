from homework2_rent import score_rent

def test_rent():
    r2_value, _, _, _ = score_rent()
    assert r2_value > 0.5

