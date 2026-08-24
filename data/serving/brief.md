# Gameweek 2 brief

**Squad cost** £100.0m  ·  **expected points** 54.3  ·  **captain** B.Fernandes (vice Haaland)

## Starting XI
```
   web_name position        team  price  expected_points
       Leno       GK      Fulham    4.5             2.95
     Virgil      DEF   Liverpool    6.5             4.00
   Truffert      DEF Bournemouth    5.5             3.90
     Senesi      DEF       Spurs    6.0             3.62
B.Fernandes      MID     Man Utd   12.0             7.97
 Szoboszlai      MID   Liverpool    7.0             4.93
  Tavernier      MID Bournemouth    6.0             3.90
       Groß      MID    Brighton    5.5             3.57
     Ndiaye      MID     Everton    6.0             3.35
    Haaland      FWD    Man City   15.5             4.46
    Watkins      FWD Aston Villa    8.0             3.71
```

## Bench
```
  web_name position         team  price  expected_points
    Justin      DEF        Leeds    4.5             3.02
Verbruggen       GK     Brighton    4.5             2.56
    O'Shea      DEF Ipswich Town    4.0             1.02
     Neave      FWD    Newcastle    4.5             0.52
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name        team  expected_points  captain_points  cost_vs_best
B.Fernandes     Man Utd             7.97           15.95          0.00
 Szoboszlai   Liverpool             4.93            9.87          6.08
    Haaland    Man City             4.46            8.91          7.04
     Virgil   Liverpool             4.00            8.00          7.95
   Truffert Bournemouth             3.90            7.80          8.15
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost         7.1
triple captain      8.0
```

## Availability risk
Squad members the minutes model rates below 70% to appear.
```
web_name position      team  p_appear  expected_minutes
   Neave      FWD Newcastle      0.27              8.23
 Haaland      FWD  Man City      0.69             51.91
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.