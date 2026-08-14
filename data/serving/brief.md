# Gameweek 1 brief

**Squad cost** £100.0m  ·  **expected points** 48.2  ·  **captain** B.Fernandes (vice Szoboszlai)

## Starting XI
```
   web_name position        team  price  expected_points
   Pickford       GK     Everton    5.5             3.20
  Tarkowski      DEF     Everton    6.0             3.72
     Virgil      DEF   Liverpool    6.5             3.31
     Senesi      DEF       Spurs    6.0             3.13
B.Fernandes      MID     Man Utd   12.0             5.28
   Anderson      MID    Man City    6.5             4.13
 Szoboszlai      MID   Liverpool    7.0             4.11
     Mbeumo      MID     Man Utd    8.0             3.83
      Gakpo      MID   Liverpool    7.0             3.63
     Thiago      FWD   Brentford    8.0             4.43
    Watkins      FWD Aston Villa    8.0             4.11
```

## Bench
```
  web_name position      team  price  expected_points
     Osula      FWD Newcastle    6.0             3.30
      Leno       GK    Fulham    4.5             2.88
  Robinson      DEF    Fulham    4.5             2.84
F.Kadıoğlu      DEF  Brighton    4.5             2.90
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name        team  expected_points  captain_points  cost_vs_best
B.Fernandes     Man Utd             5.28           10.55          0.00
     Thiago   Brentford             4.43            8.86          1.69
   Anderson    Man City             4.13            8.27          2.29
 Szoboszlai   Liverpool             4.11            8.22          2.33
    Watkins Aston Villa             4.11            8.21          2.34
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        11.9
triple captain      5.3
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.