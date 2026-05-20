from order_constraints import normalize_entry_qty, normalize_reduce_only_qty


ETH_INFO = {
    "lotSizeFilter": {
        "minOrderQty": "0.01",
        "qtyStep": "0.01",
        "maxMktOrderQty": "1600.00",
        "minNotionalValue": "5",
    }
}

BTC_INFO = {
    "lotSizeFilter": {
        "minOrderQty": "0.001",
        "qtyStep": "0.001",
        "maxMktOrderQty": "119.000",
        "minNotionalValue": "5",
    }
}


def test_entry_qty_rejects_undersized_order_without_adjustment():
    result = normalize_entry_qty(
        desired_qty=2.0 / 2065.24,
        price=2065.24,
        instrument_info=ETH_INFO,
        allow_increase=False,
    )
    assert result["ok"] is False
    assert result["reason"] == "below_exchange_minimum"
    assert abs(result["min_required_qty"] - 0.01) < 1e-9


def test_entry_qty_adjusts_to_exchange_minimum():
    result = normalize_entry_qty(
        desired_qty=2.0 / 2065.24,
        price=2065.24,
        instrument_info=ETH_INFO,
        allow_increase=True,
    )
    assert result["ok"] is True
    assert abs(result["qty"] - 0.01) < 1e-9
    assert result["qty_str"] == "0.01"


def test_entry_qty_obeys_btc_min_qty_even_when_notional_is_small():
    result = normalize_entry_qty(
        desired_qty=10.0 / 68397.19,
        price=68397.19,
        instrument_info=BTC_INFO,
        allow_increase=True,
    )
    assert result["ok"] is True
    assert abs(result["qty"] - 0.001) < 1e-9
    assert result["effective_notional"] >= 5.0


def test_reduce_only_promotes_tiny_partial_to_full_close():
    result = normalize_reduce_only_qty(
        desired_qty=0.005,
        remaining_qty=0.01,
        instrument_info=ETH_INFO,
    )
    assert result["ok"] is True
    assert abs(result["qty"] - 0.01) < 1e-9
    assert result["promoted_to_full_close"] is True


if __name__ == "__main__":
    test_entry_qty_rejects_undersized_order_without_adjustment()
    test_entry_qty_adjusts_to_exchange_minimum()
    test_entry_qty_obeys_btc_min_qty_even_when_notional_is_small()
    test_reduce_only_promotes_tiny_partial_to_full_close()
    print("test_order_constraints.py: OK")
