# Gameweek 2 brief

**Squad cost** £100.0m  ·  **expected points** 65.3  ·  **captain** B.Fernandes (vice Haaland)

## Starting XI
```
     web_name position        team  price  expected_points
     Tzolakis       GK   Hull City    4.5             4.34
      Gabriel      DEF     Arsenal    8.0             4.59
        Guéhi      DEF    Man City    6.0             4.53
       Botman      DEF   Newcastle    5.0             3.71
  B.Fernandes      MID     Man Utd   12.0             8.49
       Mbeumo      MID     Man Utd    8.0             6.63
   Szoboszlai      MID   Liverpool    7.0             5.16
    M.Sangaré      MID   Brentford    5.5             4.50
    Tavernier      MID Bournemouth    6.0             4.39
      Haaland      FWD    Man City   15.5             6.15
Calvert-Lewin      FWD       Leeds    6.0             4.26
```

## Bench
```
   web_name position         team  price  expected_points
       Egan      DEF    Hull City    4.0             3.61
     O'Shea      DEF Ipswich Town    4.0             2.94
Walle Egeli      FWD Ipswich Town    4.5             0.78
   Dubravka       GK        Spurs    4.0             0.37
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
B.Fernandes   Man Utd             8.49           16.98          0.00
     Mbeumo   Man Utd             6.63           13.26          3.72
    Haaland  Man City             6.15           12.30          4.68
 Szoboszlai Liverpool             5.16           10.32          6.66
    Gabriel   Arsenal             4.59            9.17          7.81
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost         7.7
triple captain      8.5
```

## Availability risk
Squad members the minutes model rates below 70% to appear.
```
   web_name position         team  p_appear  expected_minutes
   Dubravka       GK        Spurs      0.16             12.37
Walle Egeli      FWD Ipswich Town      0.50             16.32
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.