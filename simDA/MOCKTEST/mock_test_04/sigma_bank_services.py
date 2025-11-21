class BankingAPI:
    @staticmethod
    def get_balance(user_id):
        # Simulate fetching the balance from an external system
        # In a real implementation, this would call a real API
        # Example:
        # return requests.get(f"https://api.bank.com/users/{user_id}/balance")
        pass

    @staticmethod
    def initiate_transfer(sender, receiver, amount):
        # Simulate initiating a money transfer in an external system
        # In a real implementation, this would call a real API
        # Example:
        # return requests.post(f"https://api.bank.com/users/{sender}/transfer",
        # data={"receiver": receiver, "amount": amount})
        pass


class UserNotFoundException(Exception):
    pass


class InsufficientBalanceException(Exception):
    pass


# Some code that uses the BankingAPI
# We want to test this code
def check_balance(user_id):
    # Call the BankingAPI to get the balance
    return BankingAPI.get_balance(user_id)


def transfer_money(sender, receiver, amount):
    # Call the BankingAPI to initiate a money transfer
    return BankingAPI.initiate_transfer(sender, receiver, amount)
