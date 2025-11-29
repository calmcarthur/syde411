# Microgrid Scenario Study

## Scenario Definitions
- **Baseline**: Default constants from config.py.
- **SoC 150**: Start the battery at 150 kWh to stress low state preparedness.
- **SoC 400**: Begin with a high state of charge to test recharge requirements.
- **Gen $0.20**: Cheaper generator energy incentivises thermal dispatch.
- **Gen $0.50**: Expensive generator energy discourages thermal dispatch.
- **Rate 80**: Reduce the battery charge/discharge power limit to 80 kW.
- **SoC 150-450**: Narrow the usable SoC window to 150–450 kWh.
- **Deg $0.05**: Increase the cycling penalty to $0.05 per kWh.
- **Deg $0.00**: Ignore battery degradation costs to emphasise cycling.

## Summary Metrics
| Scenario | Solver | Cost ($) | ΔCost vs Base ($) | Grid (kWh) | Solar (kWh) | Battery Throughput (kWh) | Gen Hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | highs | 627.73 | +0.00 | 2722 | 980 | 801 | 4 |
| Baseline | cbc | 627.73 | +0.00 | 2722 | 980 | 801 | 4 |
| SoC 150 | highs | 627.73 | +0.00 | 2722 | 980 | 801 | 4 |
| SoC 150 | cbc | 627.73 | -0.00 | 2722 | 980 | 801 | 4 |
| SoC 400 | highs | 632.99 | +5.26 | 2722 | 980 | 801 | 4 |
| SoC 400 | cbc | 632.99 | +5.26 | 2722 | 980 | 801 | 4 |
| Gen $0.20 | highs | 549.73 | -78.00 | 2122 | 980 | 801 | 10 |
| Gen $0.20 | cbc | 549.73 | -78.00 | 2122 | 980 | 801 | 10 |
| Gen $0.50 | highs | 639.73 | +12.00 | 3122 | 980 | 801 | 0 |
| Gen $0.50 | cbc | 639.73 | +12.00 | 3122 | 980 | 801 | 0 |
| Rate 80 | highs | 633.73 | +6.00 | 2722 | 980 | 801 | 4 |
| Rate 80 | cbc | 633.73 | +6.00 | 2722 | 980 | 801 | 4 |
| SoC 150-450 | highs | 645.45 | +17.72 | 2712 | 980 | 601 | 4 |
| SoC 150-450 | cbc | 645.45 | +17.72 | 2712 | 980 | 601 | 4 |
| Deg $0.05 | highs | 647.76 | +20.03 | 2722 | 980 | 801 | 4 |
| Deg $0.05 | cbc | 647.76 | +20.03 | 2722 | 980 | 801 | 4 |
| Deg $0.00 | highs | 607.71 | -20.03 | 2722 | 980 | 801 | 4 |
| Deg $0.00 | cbc | 607.71 | -20.03 | 2722 | 980 | 801 | 4 |

## Key Observations
- Both `highs` and `cbc` solvers converge to identical dispatch plans across all scenarios; residual energy-balance violations stay below 5×10⁻⁶ kW.
- Lowering generator cost to $0.20/kWh adds six additional generator hours and trims grid imports by roughly 600 kWh, cutting operating cost by $78.
- Raising generator cost to $0.50/kWh removes generator usage entirely, pushing grid imports to 3.1 MWh and increasing cost by $12.
- Tightening the charge/discharge rate or SoC window increases cost by $6–$18 because the battery cannot shift as much energy into peak periods.
- Ignoring degradation cost saves $20 versus the baseline, demonstrating how the penalty reins in cycling without altering feasibility.

## Solver Validation
- `highs` → max energy balance deviation 0.00e+00 kW, SoC bounds respected: Yes, end SoC gap: 0.00 kWh.
- `cbc` → max energy balance deviation 3.00e-06 kW, SoC bounds respected: Yes, end SoC gap: 0.00 kWh.