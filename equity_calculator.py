import eval7

def parse_into_pokerstove(card_strs):
    """
    Converts an array of RLCard style card strings to PokerStove style card strings.
    
    Args:
    - card_strs: A list of strings representing cards in RLCard format
    
    Returns:
    - A list of strings representing cards in PokerStove format
    """
    # Mapping from RLCard suit to PokerStove suit
    suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
    # Mapping for ranks that are the same in both formats
    rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
                '7': '7', '8': '8', '9': '9', 'T': 'T', 'J': 'J',
                'Q': 'Q', 'K': 'K', 'A': 'A'}
    
    pokerstove_cards = []
    for card in card_strs:
        # Extract the rank and suit from the RLCard card string
        suit, rank = card[0], card[1]
        # Convert to PokerStove format and add to the result list
        pokerstove_cards.append(rank_map[rank] + suit_map[suit].lower())
    
    return pokerstove_cards

def rl_cardstr_to_cards(card_strs):
    """
    Converts an array of card strings from RLCard format to a list of `Card` objects suitable for use with the eval7 library.
   
    Parameters:
    - card_strs (list of str): A list of strings representing cards in RLCard format. Each string should be exactly two characters long,
      where the first character denotes the rank and the second character denotes the suit.
    
    Returns:
    - iterator of eval7.Card: An iterator over `Card` objects corresponding to the input card strings. Each `Card` object represents
      a card in the format expected by the eval7 library, which can be used for evaluating poker hands.
    
   
    Example:
        # Converting RLCard format strings to eval7 Card objects
        rl_cards = ["HA", "D2", "CT", "SJ"]
        eval7_cards = rl_cardstr_to_cards(rl_cards)
        for card in eval7_cards:
            print(card)  # Prints the eval7 representation of each card
        
    Note:
    This function depends on the `parse_into_pokerstove` function to convert RLCard format strings to PokerStove-style strings.
    Ensure that `parse_into_pokerstove` is correctly implemented and available in the scope.
    """
    pokerstove_strings = parse_into_pokerstove(card_strs)
    cards = tuple(map(eval7.Card, pokerstove_strings))
    return cards


def calculate_equity(hole_cards, community_cards):
    """
    Given hole_cards and community_cards array in RLCard format, return win percentage vs. all hands.

    """
    hero = eval7.HandRange("".join(parse_into_pokerstove(hole_cards)))
    
    
    villain = eval7.HandRange("22+, A2+, K2+, Q2+, J2+, T2+, 92+, 82+, 72+, 62+, 52+, 42+, 32+")
    board = rl_cardstr_to_cards(community_cards)
    equity_map = eval7.py_all_hands_vs_range(hero, villain, board, 100000)
    return equity_map[next(iter(equity_map))]

# some tests

# Tests for parse_into_pokerstove
def test_parse_into_pokerstove_basic():
    input_cards = ["SA", "HK", "DQ", "CJ"]
    expected = ["As", "Kh", "Qd", "Jc"]
    assert parse_into_pokerstove(input_cards) == expected, "Basic conversion failed."

# Test for rl_cardstr_to_cards (implicitly tests parse_into_pokerstove)
def test_rl_cardstr_to_cards_integration():
    input_cards = ["SA", "HK", "DQ", "CJ"]
    expected_ranks = ["A", "K", "Q", "J"]
    cards = rl_cardstr_to_cards(input_cards)
    for card, expected_rank in zip(cards, expected_ranks):
        assert str(card)[0] == expected_rank, f"Card rank {expected_rank} not found in the output."

# Tests for calculate_equity
def test_calculate_equity_simple_scenario():
    hole_cards = ["SA", "HA"]  # Strong hand
    community_cards = []
    win_percentage = calculate_equity(hole_cards, community_cards)
    assert win_percentage > 0.8, "Expected high win percentage for AA."

def test_calculate_equity_community_cards_impact():
    hole_cards = ["SA", "HQ"]
    community_cards = ["D2", "S3", "S4", "S5", "D6"]  # Straight flush on board
    win_percentage = calculate_equity(hole_cards, community_cards)
    assert win_percentage < 0.8, "Expected lower win percentage due to strong community cards."

# Running the tests
def main():
    test_parse_into_pokerstove_basic()
    test_rl_cardstr_to_cards_integration()
    test_calculate_equity_simple_scenario()
    test_calculate_equity_community_cards_impact()
    
    print("All tests passed successfully!")
    

if __name__ == "__main__":
    main()

