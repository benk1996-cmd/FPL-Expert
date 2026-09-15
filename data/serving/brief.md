# Gameweek 5 brief

**Squad cost** £100.0m  ·  **expected points** 63.0  ·  **captain** Haaland (vice B.Fernandes)

## Starting XI
```
   web_name position          team  price  expected_points
   Tzolakis       GK     Hull City    4.6             4.35
      Guéhi      DEF      Man City    6.0             4.90
       Egan      DEF     Hull City    4.1             4.27
     Botman      DEF     Newcastle    5.0             3.98
Gibbs-White      MID Nott'm Forest    7.9             6.56
       Saka      MID       Arsenal    9.5             5.38
B.Fernandes      MID       Man Utd   12.0             5.38
   Anderson      MID      Man City    6.3             4.90
      Scott      MID   Bournemouth    6.1             4.32
    Haaland      FWD      Man City   15.5             6.71
      Barry      FWD       Everton    5.6             5.51
```

## Bench
```
 web_name position      team  price  expected_points
    Giles      DEF Hull City    4.0             3.39
     Leno       GK    Fulham    4.5             2.99
Robertson      DEF     Spurs    4.4             2.55
   Mheuka      FWD   Chelsea    4.5             0.21
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name          team  expected_points  captain_points  cost_vs_best
    Haaland      Man City             6.71           13.43          0.00
Gibbs-White Nott'm Forest             6.56           13.13          0.30
      Barry       Everton             5.51           11.02          2.41
       Saka       Arsenal             5.38           10.75          2.67
B.Fernandes       Man Utd             5.38           10.75          2.68
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost         9.1
triple captain      6.7
```

## Availability risk
Squad members the minutes model rates below 70% to appear.
```
web_name position    team  p_appear  expected_minutes
  Mheuka      FWD Chelsea      0.06              1.67
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.