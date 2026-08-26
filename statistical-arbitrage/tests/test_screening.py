from src.data_loader import load_prices, train_test_split
from src.screening import screen_pairs


def test_screening_ranks_visa_mastercard_as_most_cointegrated():
    prices = load_prices()
    train, _test = train_test_split(prices)
    candidates = [("V", "MA"), ("KO", "PEP"), ("XOM", "CVX")]
    results = screen_pairs(train, candidates)
    assert results.iloc[0]["pair"] == "V-MA"
    assert results.iloc[0]["cointegrated"]
