# Gameweek 5 brief

**Squad cost** £99.8m  ·  **expected points** 50.2  ·  **captain** B.Fernandes (vice Guéhi)

## Your team as it stands
No transfer made. Expected **53.0** points this gameweek, captain included.
```
     web_name position      team  price  expected_points
     Pickford       GK   Everton    5.5             3.59
    Tarkowski      DEF   Everton    6.1             4.89
         Egan      DEF Hull City    4.1             4.27
       Virgil      DEF Liverpool    6.5             3.51
  B.Fernandes      MID   Man Utd   12.0             5.38
   Szoboszlai      MID Liverpool    7.0             4.61
       Mbeumo      MID   Man Utd    7.9             4.38
       Rogers      MID   Chelsea    7.7             3.73
       Thiago      FWD Brentford    7.9             4.64
   João Pedro      FWD   Chelsea    7.8             4.42
Calvert-Lewin      FWD     Leeds    6.0             4.24
```
*bench*
```
  web_name position      team  price  expected_points
     Gakpo      MID Liverpool    7.2             3.15
      Leno       GK    Fulham    4.5             2.99
  Robinson      DEF    Fulham    4.5             2.84
F.Kadıoğlu      DEF  Brighton    4.4             2.31
```

## Transfers
**2 transfer(s)**, 1 hit(s) costing 4  ·  net gain **+11.64** points over the horizon
```
OUT
  web_name position      team  selling_price  horizon_points
F.Kadıoğlu      DEF  Brighton            4.4           10.90
     Gakpo      MID Liverpool            7.1           12.48

IN
web_name position     team  price  horizon_points
   Guéhi      DEF Man City    6.0           18.52
Anderson      MID Man City    6.3           20.50
```

## What the move buys
```
this gameweek       53.04  ->   55.61   +2.57
over the horizon                        +15.64   net +11.64 after a 4pt hit
```
Changes the XI: **Anderson, Guéhi** in, **Rogers, Virgil** out.
The horizon margin behind a transfer is measurably overstated — a fit of realised on forecast gain across 111 decisions gives a slope of 0.436, stable in every season. Treat a small positive as closer to zero than it reads.

## Starting XI — after the transfer
```
     web_name position      team  price  expected_points
     Pickford       GK   Everton    5.5             3.59
        Guéhi      DEF  Man City    6.0             4.90
    Tarkowski      DEF   Everton    6.1             4.89
         Egan      DEF Hull City    4.1             4.27
  B.Fernandes      MID   Man Utd   12.0             5.38
     Anderson      MID  Man City    6.3             4.90
   Szoboszlai      MID Liverpool    7.0             4.61
       Mbeumo      MID   Man Utd    7.9             4.38
       Thiago      FWD Brentford    7.9             4.64
   João Pedro      FWD   Chelsea    7.8             4.42
Calvert-Lewin      FWD     Leeds    6.0             4.24
```

## Bench
```
web_name position      team  price  expected_points
  Rogers      MID   Chelsea    7.7             3.73
  Virgil      DEF Liverpool    6.5             3.51
    Leno       GK    Fulham    4.5             2.99
Robinson      DEF    Fulham    4.5             2.84
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
B.Fernandes   Man Utd             5.38           10.75          0.00
      Guéhi  Man City             4.90            9.81          0.94
   Anderson  Man City             4.90            9.81          0.94
  Tarkowski   Everton             4.89            9.78          0.97
     Thiago Brentford             4.64            9.29          1.46
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        13.1
triple captain      5.4
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.