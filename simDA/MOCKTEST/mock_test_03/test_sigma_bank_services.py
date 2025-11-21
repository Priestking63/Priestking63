from unittest.mock import Mock
from unittest.mock import patch

from sigma_bank_services import check_balance
from sigma_bank_services import get_transaction_history
from sigma_bank_services import transfer_money

# Create a mock object for the BankingAPI
mock_api = Mock()


def test_check_balance():
    # Patch the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance estimated return value
        mock_api.get_balance.return_value = 1500

        # Test the check_balance function
        balance = check_balance("User123")

        # check result and method was called
        assert balance == 1500
        assert mock_api.get_balance.call_count == 1
        mock_api.reset_mock()


def test_transfer_money():
    # Patch the the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance estimated return value
        mock_api.get_balance.return_value = 1500
        # Set the method initiate_transfer estimated return value
        mock_api.initiate_transfer.return_value = "Transfer Successful"

        # Test the transfer_money function
        status = transfer_money("User123", "User456", 200)

        # check result and methods were called
        assert status == "Transfer Successful"
        assert mock_api.get_balance.call_count == 1
        assert mock_api.initiate_transfer.call_count == 1
        mock_api.reset_mock()


def test_get_transaction_history():
    # Patch the the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method fetch_transactions estimated return value
        mock_api.fetch_transactions.return_value = [
            "Transfer to User456: 200",
            "Deposit from User789: 500",
        ]
        # Set the method get_balance estimated return value
        mock_api.get_balance.return_value = 1800

        # Test the get_transaction_history function
        history = get_transaction_history("User123")

        # check result and methods were called
        assert history == [
            "Transfer to User456: 200",
            "Deposit from User789: 500",
            "Balance: 1800",
        ]
        assert mock_api.fetch_transactions.call_count == 1
        assert mock_api.get_balance.call_count == 1.
        mock_api.reset_mock()
