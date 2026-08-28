# Gameweek 2 brief

**Squad cost** £99.5m  ·  **expected points** 51.9  ·  **captain** B.Fernandes (vice Mbeumo)

## Your team as it stands
No transfer made. Expected **57.8** points this gameweek, captain included.
```
   web_name position      team  price  expected_points
       Leno       GK    Fulham    4.5             3.20
     Virgil      DEF Liverpool    6.5             4.17
     Senesi      DEF     Spurs    6.0             3.95
  Tarkowski      DEF   Everton    6.0             3.31
B.Fernandes      MID   Man Utd   12.0             8.49
     Mbeumo      MID   Man Utd    8.0             6.63
 Szoboszlai      MID Liverpool    7.0             5.16
      Gakpo      MID Liverpool    7.0             4.35
   Anderson      MID  Man City    6.5             3.04
     Thiago      FWD Brentford    8.0             3.81
      Osula      FWD Newcastle    6.0             3.14
```
*bench*
```
  web_name position        team  price  expected_points
  Pickford       GK     Everton    5.5             2.91
  Robinson      DEF      Fulham    4.5             2.83
F.Kadıoğlu      DEF    Brighton    4.5             0.61
   Watkins      FWD Aston Villa    8.0             0.00
```

## Transfers
**2 transfer(s)**, 1 hit(s) costing 4  ·  net gain **+28.98** points over the horizon
```
OUT
  web_name position        team  selling_price  horizon_points
   Watkins      FWD Aston Villa            8.0            0.00
F.Kadıoğlu      DEF    Brighton            4.5            2.86

IN
     web_name position     team  price  horizon_points
Calvert-Lewin      FWD    Leeds    6.0           16.28
        Guéhi      DEF Man City    6.0           19.56
```

## What the move buys
```
this gameweek       57.75  ->   60.36   +2.61
over the horizon                        +32.98   net +28.98 after a 4pt hit
```
Changes the XI: **Calvert-Lewin, Guéhi** in, **Anderson, Osula** out.
The horizon margin behind a transfer is measurably overstated — a fit of realised on forecast gain across 111 decisions gives a slope of 0.436, stable in every season. Treat a small positive as closer to zero than it reads.

## Starting XI — after the transfer
```
     web_name position      team  price  expected_points
         Leno       GK    Fulham    4.5             3.20
        Guéhi      DEF  Man City    6.0             4.53
       Virgil      DEF Liverpool    6.5             4.17
       Senesi      DEF     Spurs    6.0             3.95
    Tarkowski      DEF   Everton    6.0             3.31
  B.Fernandes      MID   Man Utd   12.0             8.49
       Mbeumo      MID   Man Utd    8.0             6.63
   Szoboszlai      MID Liverpool    7.0             5.16
        Gakpo      MID Liverpool    7.0             4.35
Calvert-Lewin      FWD     Leeds    6.0             4.26
       Thiago      FWD Brentford    8.0             3.81
```

## Bench
```
web_name position      team  price  expected_points
   Osula      FWD Newcastle    6.0             3.14
Anderson      MID  Man City    6.5             3.04
Pickford       GK   Everton    5.5             2.91
Robinson      DEF    Fulham    4.5             2.83
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
B.Fernandes   Man Utd             8.49           16.98          0.00
     Mbeumo   Man Utd             6.63           13.26          3.72
 Szoboszlai Liverpool             5.16           10.32          6.66
      Guéhi  Man City             4.53            9.06          7.92
      Gakpo Liverpool             4.35            8.71          8.27
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        11.9
triple captain      8.5
```

## Availability risk
Squad members the minutes model rates below 70% to appear.
```
web_name position     team  p_appear  expected_minutes
Anderson      MID Man City      0.66              49.8
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.