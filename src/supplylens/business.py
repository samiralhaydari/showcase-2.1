def invoice_cost(foreign_amount, myr_per_foreign_unit):
    if foreign_amount < 0 or myr_per_foreign_unit <= 0:
        raise ValueError("Amount must be non-negative and rate must be positive.")
    return float(foreign_amount * myr_per_foreign_unit)


def landed_cost(invoice_myr, freight_myr=0, insurance_myr=0, duties_myr=0, other_myr=0):
    values = [invoice_myr, freight_myr, insurance_myr, duties_myr, other_myr]
    if any(v < 0 for v in values):
        raise ValueError("Cost inputs cannot be negative.")
    return float(sum(values))


def gross_margin(selling_price, imported_material_cost, local_costs):
    if selling_price <= 0:
        raise ValueError("Selling price must be greater than zero.")
    total_cost = imported_material_cost + local_costs
    gross_profit = selling_price - total_cost
    return {
        "total_cost": float(total_cost),
        "gross_profit": float(gross_profit),
        "gross_margin_pct": float(gross_profit / selling_price * 100),
    }


def apply_user_scenario_shock(rate, shock_pct):
    # User-defined scenario only; it is not a model prediction.
    return float(rate * (1 + shock_pct / 100))
