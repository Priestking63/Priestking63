from unittest.mock import patch

from currency import get_exchange_rate


@patch("currency.requests.get")
def test_get_exchange_rate(mock_get):
    '''mock'''
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {'rate': 1.2}

    # Call the function
    result = get_exchange_rate("USD", "EUR")

    # Assert the result
    assert result == 1.2
