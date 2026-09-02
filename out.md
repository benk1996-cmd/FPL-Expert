# Gameweek 3 brief

**Squad cost** £96.5m  ·  **expected points** 52.9  ·  **captain** Szoboszlai (vice Egan)

## Your team as it stands
No transfer made. Expected **55.4** points this gameweek, captain included.
```
     web_name position      team  price  expected_points
         Leno       GK    Fulham    4.5             3.41
       Virgil      DEF Liverpool    6.5             4.40
    Tarkowski      DEF   Everton    6.0             3.70
     Robinson      DEF    Fulham    4.5             3.51
   Szoboszlai      MID Liverpool    7.0             6.43
     Anderson      MID  Man City    6.4             5.33
  B.Fernandes      MID   Man Utd   12.0             5.28
        Gakpo      MID Liverpool    7.0             4.76
       Mbeumo      MID   Man Utd    8.0             4.24
       Thiago      FWD Brentford    8.0             4.16
Calvert-Lewin      FWD     Leeds    6.0             3.74
```
*bench*
```
  web_name position     team  price  expected_points
João Pedro      FWD  Chelsea    7.7             3.10
  Pickford       GK  Everton    5.5             3.08
F.Kadıoğlu      DEF Brighton    4.4             3.05
    Senesi      DEF    Spurs    5.9             2.52
```

## Transfers
**2 transfer(s)**, 1 hit(s) costing 4  ·  net gain **+15.91** points over the horizon
```
OUT
web_name position    team  selling_price  horizon_points
Pickford       GK Everton            5.5           12.79
  Senesi      DEF   Spurs            5.9           11.28

IN
web_name position      team  price  horizon_points
    Egan      DEF Hull City    4.0           22.49
Tzolakis       GK Hull City    4.5           21.50
```

## What the move buys
```
this gameweek       55.39  ->   59.33   +3.94
over the horizon                        +19.91   net +15.91 after a 4pt hit
```
Changes the XI: **Egan, Tzolakis** in, **Leno, Robinson** out.
The horizon margin behind a transfer is measurably overstated — a fit of realised on forecast gain across 111 decisions gives a slope of 0.436, stable in every season. Treat a small positive as closer to zero than it reads.

## Starting XI — after the transfer
```
     web_name position      team  price  expected_points
     Tzolakis       GK Hull City    4.5             5.36
         Egan      DEF Hull City    4.0             5.50
       Virgil      DEF Liverpool    6.5             4.40
    Tarkowski      DEF   Everton    6.0             3.70
   Szoboszlai      MID Liverpool    7.0             6.43
     Anderson      MID  Man City    6.4             5.33
  B.Fernandes      MID   Man Utd   12.0             5.28
        Gakpo      MID Liverpool    7.0             4.76
       Mbeumo      MID   Man Utd    8.0             4.24
       Thiago      FWD Brentford    8.0             4.16
Calvert-Lewin      FWD     Leeds    6.0             3.74
```

## Bench
```
  web_name position     team  price  expected_points
  Robinson      DEF   Fulham    4.5             3.51
      Leno       GK   Fulham    4.5             3.41
João Pedro      FWD  Chelsea    7.7             3.10
F.Kadıoğlu      DEF Brighton    4.4             3.05
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
 Szoboszlai Liverpool             6.43           12.86          0.00
       Egan Hull City             5.50           11.00          1.86
   Tzolakis Hull City             5.36           10.73          2.14
   Anderson  Man City             5.33           10.66          2.21
B.Fernandes   Man Utd             5.28           10.56          2.30
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        13.1
triple captain      6.4
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.