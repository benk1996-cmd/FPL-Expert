# Gameweek 4 brief

**Squad cost** £99.1m  ·  **expected points** 49.0  ·  **captain** Szoboszlai (vice B.Fernandes)

## Your team as it stands
No transfer made. Expected **52.8** points this gameweek, captain included.
```
     web_name position      team  price  expected_points
     Pickford       GK   Everton    5.5             3.15
         Egan      DEF Hull City    4.1             4.85
       Virgil      DEF Liverpool    6.5             4.39
    Tarkowski      DEF   Everton    6.0             3.94
   Szoboszlai      MID Liverpool    7.0             5.36
  B.Fernandes      MID   Man Utd   12.0             4.95
        Gakpo      MID Liverpool    7.2             4.63
     Anderson      MID  Man City    6.3             4.34
       Mbeumo      MID   Man Utd    7.9             4.01
       Thiago      FWD Brentford    7.9             3.93
Calvert-Lewin      FWD     Leeds    6.0             3.86
```
*bench*
```
  web_name position     team  price  expected_points
F.Kadıoğlu      DEF Brighton    4.4             3.57
João Pedro      FWD  Chelsea    7.7             2.97
      Leno       GK   Fulham    4.5             2.62
  Robinson      DEF   Fulham    4.5             1.94
```

## Transfers
**2 transfer(s)**, 1 hit(s) costing 4  ·  net gain **+13.54** points over the horizon
```
OUT
web_name position   team  selling_price  horizon_points
    Leno       GK Fulham            4.5           12.36
Robinson      DEF Fulham            4.5           10.27

IN
web_name position      team  price  horizon_points
   Guéhi      DEF  Man City    6.0           18.09
Tzolakis       GK Hull City    4.6           22.09
```

## What the move buys
```
this gameweek       52.77  ->   54.34   +1.58
over the horizon                        +17.54   net +13.54 after a 4pt hit
```
Changes the XI: **Tzolakis** in, **Pickford** out.
The horizon margin behind a transfer is measurably overstated — a fit of realised on forecast gain across 111 decisions gives a slope of 0.436, stable in every season. Treat a small positive as closer to zero than it reads.

## Starting XI — after the transfer
```
     web_name position      team  price  expected_points
     Tzolakis       GK Hull City    4.6             4.72
         Egan      DEF Hull City    4.1             4.85
       Virgil      DEF Liverpool    6.5             4.39
    Tarkowski      DEF   Everton    6.0             3.94
   Szoboszlai      MID Liverpool    7.0             5.36
  B.Fernandes      MID   Man Utd   12.0             4.95
        Gakpo      MID Liverpool    7.2             4.63
     Anderson      MID  Man City    6.3             4.34
       Mbeumo      MID   Man Utd    7.9             4.01
       Thiago      FWD Brentford    7.9             3.93
Calvert-Lewin      FWD     Leeds    6.0             3.86
```

## Bench
```
  web_name position     team  price  expected_points
     Guéhi      DEF Man City    6.0             3.77
F.Kadıoğlu      DEF Brighton    4.4             3.57
  Pickford       GK  Everton    5.5             3.15
João Pedro      FWD  Chelsea    7.7             2.97
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
 Szoboszlai Liverpool             5.36           10.72          0.00
B.Fernandes   Man Utd             4.95            9.89          0.82
       Egan Hull City             4.85            9.71          1.01
   Tzolakis Hull City             4.72            9.45          1.27
      Gakpo Liverpool             4.63            9.26          1.46
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        13.5
triple captain      5.4
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.