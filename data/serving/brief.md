# Gameweek 3 brief

**Squad cost** £100.0m  ·  **expected points** 79.1  ·  **captain** Haaland (vice Tzolakis)

## Starting XI
```
   web_name position      team  price  expected_points
   Tzolakis       GK Hull City    4.5             6.80
      Guéhi      DEF  Man City    6.0             7.33
       Egan      DEF Hull City    4.0             6.68
      Ajayi      DEF Hull City    4.1             6.42
      Dedić      DEF Newcastle    4.5             4.35
 Szoboszlai      MID Liverpool    7.0             6.56
       Saka      MID   Arsenal    9.5             6.01
  M.Sangaré      MID Brentford    5.7             5.61
B.Fernandes      MID   Man Utd   12.0             5.31
    Haaland      FWD  Man City   15.5             8.65
       Isak      FWD Liverpool    9.0             6.77
```

## Bench
```
   web_name position         team  price  expected_points
  De Cuyper      DEF     Brighton    4.7             3.75
 Verbruggen       GK     Brighton    4.5             3.21
   Yalcouyé      MID     Brighton    4.5             3.12
Walle Egeli      FWD Ipswich Town    4.5             0.41
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
web_name      team  expected_points  captain_points  cost_vs_best
 Haaland  Man City             8.65           17.30          0.00
   Guéhi  Man City             7.33           14.65          2.65
Tzolakis Hull City             6.80           13.60          3.70
    Isak Liverpool             6.77           13.54          3.76
    Egan Hull City             6.68           13.37          3.93
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        10.5
triple captain      8.6
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