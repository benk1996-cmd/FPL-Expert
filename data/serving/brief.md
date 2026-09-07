# Gameweek 3 brief

**Squad cost** £100.0m  ·  **expected points** 69.5  ·  **captain** Haaland (vice B.Fernandes)

## Starting XI
```
   web_name position          team  price  expected_points
   Tzolakis       GK     Hull City    4.6             5.39
      Guéhi      DEF      Man City    6.0             5.92
       Egan      DEF     Hull City    4.0             5.45
    Gabriel      DEF       Arsenal    8.0             5.19
      Ajayi      DEF     Hull City    4.1             5.11
 Szoboszlai      MID     Liverpool    7.0             6.44
Gibbs-White      MID Nott'm Forest    7.9             5.29
B.Fernandes      MID       Man Utd   12.0             5.28
   Anderson      MID      Man City    6.4             5.17
       Groß      MID      Brighton    5.5             4.57
    Haaland      FWD      Man City   15.5             7.83
```

## Bench
```
   web_name position         team  price  expected_points
      Dedić      DEF    Newcastle    4.5             4.38
      Barry      FWD      Everton    5.5             3.20
 Verbruggen       GK     Brighton    4.5             3.21
Walle Egeli      FWD Ipswich Town    4.5             0.39
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
  web_name      team  expected_points  captain_points  cost_vs_best
   Haaland  Man City             7.83           15.65          0.00
Szoboszlai Liverpool             6.44           12.89          2.76
     Guéhi  Man City             5.92           11.84          3.81
      Egan Hull City             5.45           10.90          4.75
  Tzolakis Hull City             5.39           10.79          4.86
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        11.2
triple captain      7.8
```

## Availability risk
Squad members the minutes model rates below 70% to appear.
```
   web_name position         team  p_appear  expected_minutes
Walle Egeli      FWD Ipswich Town      0.23              7.19
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.