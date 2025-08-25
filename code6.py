from typing import Tuple

# Constants for better configurability
DISCOUNT_THRESHOLD = 100.0
DISCOUNT_RATE = 0.9  # 10% discount
DISCOUNT_MESSAGE = "Large order discount applied!"

def calculate_total(price: float, quantity: int) -> Tuple[float, bool]:
    """
    Calculates the total price for an order, applying a discount if the total exceeds a threshold.

    Args:
        price: The unit price of the item.
        quantity: The number of items.

    Returns:
        A tuple containing:
        - The final calculated total (float).
        - A boolean indicating if a discount was applied (True if yes, False if no).
    """
    initial_total = price * quantity
    discount_applied = False

    if initial_total > DISCOUNT_THRESHOLD:
        final_total = initial_total * DISCOUNT_RATE
        discount_applied = True
    else:
        final_total = initial_total

    return final_total, discount_applied

def process_order_and_print(order_num: int, price: float, quantity: int) -> None:
    """
    Processes a single order, calculates its total, and prints the result.
    """
    total, discount_applied = calculate_total(price, quantity)
    if discount_applied:
        print(DISCOUNT_MESSAGE)
    print(f"Order {order_num} total: ${total:.2f}")

def main():
    """Main function to demonstrate calculating order totals with and without discounts."""

    # Example 1: Small order
    process_order_and_print(1, 10.0, 5) # Total 50.00
    print("-" * 20)

    # Example 2: Large order
    process_order_and_print(2, 20.0, 6) # Total 120.00 -> 108.00
    print("-" * 20)

    # Example 3: Using a list for multiple orders
    orders_data = [
        {'price': 12.50, 'quantity': 3},  # Total 37.50
        {'price': 50.00, 'quantity': 2},  # Total 100.00 (at threshold, no discount)
        {'price': 50.00, 'quantity': 3},  # Total 150.00 (discounted)
    ]

    for i, order in enumerate(orders_data):
        process_order_and_print(i + 3, order['price'], order['quantity'])
        if i < len(orders_data) - 1:
            print("-" * 20)


if __name__ == "__main__":
    main()
