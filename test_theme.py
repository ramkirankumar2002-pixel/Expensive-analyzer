# test_theme.py - small tests for theme.py (plain asserts). Run with:  python test_theme.py

import categorizer as cz
import theme as th


def test_every_category_has_emoji_and_colour():
    for name in cz.CATEGORIES + ["Income", "Refund"]:
        emoji, colour = th.CATEGORY_STYLE[name]
        assert emoji and colour.startswith("#")
        assert th.label(name) in th.COLOR_MAP


def test_label_roundtrip():
    assert th.label("Food") == "🍔 Food"
    for name in th.CATEGORY_STYLE:
        assert th.unlabel(th.label(name)) == name
    assert th.unlabel("Something else") == "Something else"


def test_metric_card_colours():
    assert "delta up" in th.metric_card("💰", "Income", "₹1", 5.0)                 # income up = green
    assert "delta down" in th.metric_card("💸", "Spent", "₹1", 5.0, invert=True)   # spending up = red
    assert "delta up" in th.metric_card("💸", "Spent", "₹1", -5.0, invert=True)    # spending down = green
    assert "↑" in th.metric_card("💰", "Income", "₹1", 5.0) and "↓" in th.metric_card("💰", "Income", "₹1", -5.0)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All theme tests passed.")
