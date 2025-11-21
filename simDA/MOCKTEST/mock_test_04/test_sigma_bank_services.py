from unittest.mock import Mock
from unittest.mock import patch

from sigma_bank_services import check_balance
from sigma_bank_services import InsufficientBalanceException
from sigma_bank_services import transfer_money
from sigma_bank_services import UserNotFoundException

# Create a mock object for the BankingAPI
mock_api = Mock()


def test_check_balance_success():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance  return value 1500
        mock_api.get_balance.return_value = 1500

        # Test the check_balance function for any user_id
        
        balance = check_balance('user')
        # check result and method was called
        assert balance == 1500
        assert mock_api.get_balance.call_count == 1

        # reset mock api state
        mock_api.reset_mock()


def test_check_balance_user_not_found():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance return exception UserNotFoundException("User not found!")
        mock_api.get_balance.side_effect = UserNotFoundException("User not found!")

        # Test check balance for any user_id and assert
        try:
            check_balance("User123")
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

        # reset mock api state
        mock_api.reset_mock()


def test_transfer_money_success():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        mock_api.initiate_transfer.return_value = "Transfer Successful"

        transfer = transfer_money("User123", "user", 123)

        assert transfer == "Transfer Successful"
        mock_api.reset_mock()

def test_transfer_money_insufficient_balance():
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Implement test for transfer_money function.
        # See test_check_balance_user_not_found for reference
        mock_api.initiate_transfer.side_effect = InsufficientBalanceException("DW")

        # Test check balance for any user_id and assert
        try:
            transfer_money("User123", "user", 123)
        except InsufficientBalanceException:
            assert True
        else:
            assert False, "InsufficientBalanceException not raised"

        # reset mock api state
        mock_api.reset_mock()


def test_transfer_money_user_not_found():
    # Implement test for transfer_money function.
    # See test_check_balance_user_not_found for reference
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance return exception UserNotFoundException("User not found!")
        mock_api.initiate_transfer.side_effect = UserNotFoundException(
            "User not found!"
        )

        # Test check balance for any user_id and assert
        try:
            transfer_money("User123",'user', 123)
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

        # reset mock api state
        mock_api.reset_mock()
