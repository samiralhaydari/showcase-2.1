from src.supplylens.business import invoice_cost, landed_cost, gross_margin


def test_invoice_cost():
    assert invoice_cost(100, 4.2) == 420.0


def test_landed_cost():
    assert landed_cost(420, 10, 2, 8) == 440.0


def test_gross_margin():
    result = gross_margin(100, 35, 40)
    assert result["total_cost"] == 75
    assert result["gross_profit"] == 25
    assert result["gross_margin_pct"] == 25
