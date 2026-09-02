# Gameweek 3 brief

**Squad cost** £99.5m  ·  **expected points** 50.7  ·  **captain** Szoboszlai (vice Gakpo)

## Your team as it stands
No transfer made. Expected **55.7** points this gameweek, captain included.
```
     web_name position      team  price  expected_points
         Leno       GK    Fulham    4.5             3.41
       Virgil      DEF Liverpool    6.5             5.00
       Senesi      DEF     Spurs    6.0             3.64
    Tarkowski      DEF   Everton    6.0             3.56
   Szoboszlai      MID Liverpool    7.0             6.52
        Gakpo      MID Liverpool    7.0             5.48
  B.Fernandes      MID   Man Utd   12.0             5.15
     Anderson      MID  Man City    6.4             4.57
       Mbeumo      MID   Man Utd    8.0             4.19
       Thiago      FWD Brentford    8.0             3.96
Calvert-Lewin      FWD     Leeds    6.0             3.69
```
*bench*
```
  web_name position     team  price  expected_points
  Robinson      DEF   Fulham    4.5             3.17
  Pickford       GK  Everton    5.5             3.05
João Pedro      FWD  Chelsea    7.6             2.96
F.Kadıoğlu      DEF Brighton    4.4             1.05
```

## Transfers
**2 transfer(s)**, 1 hit(s) costing 4  ·  net gain **+13.70** points over the horizon
```
OUT
  web_name position     team  selling_price  horizon_points
F.Kadıoğlu      DEF Brighton            4.4            3.85
  Pickford       GK  Everton            5.5           12.95

IN
web_name position      team  price  horizon_points
   White      DEF   Arsenal    5.5           16.97
Tzolakis       GK Hull City    4.5           17.53
```

## What the move buys
```
this gameweek       55.69  ->   57.26   +1.57
over the horizon                        +17.70   net +13.70 after a 4pt hit
```
Changes the XI: **Tzolakis, White** in, **Leno, Tarkowski** out.
The horizon margin behind a transfer is measurably overstated — a fit of realised on forecast gain across 111 decisions gives a slope of 0.436, stable in every season. Treat a small positive as closer to zero than it reads.

## Starting XI — after the transfer
```
     web_name position      team  price  expected_points
     Tzolakis       GK Hull City    4.5             4.35
       Virgil      DEF Liverpool    6.5             5.00
        White      DEF   Arsenal    5.5             4.20
       Senesi      DEF     Spurs    6.0             3.64
   Szoboszlai      MID Liverpool    7.0             6.52
        Gakpo      MID Liverpool    7.0             5.48
  B.Fernandes      MID   Man Utd   12.0             5.15
     Anderson      MID  Man City    6.4             4.57
       Mbeumo      MID   Man Utd    8.0             4.19
       Thiago      FWD Brentford    8.0             3.96
Calvert-Lewin      FWD     Leeds    6.0             3.69
```

## Bench
```
  web_name position    team  price  expected_points
 Tarkowski      DEF Everton    6.0             3.56
      Leno       GK  Fulham    4.5             3.41
  Robinson      DEF  Fulham    4.5             3.17
João Pedro      FWD Chelsea    7.6             2.96
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
   web_name      team  expected_points  captain_points  cost_vs_best
 Szoboszlai Liverpool             6.52           13.04          0.00
      Gakpo Liverpool             5.48           10.95          2.08
B.Fernandes   Man Utd             5.15           10.30          2.74
     Virgil Liverpool             5.00           10.00          3.04
   Anderson  Man City             4.57            9.15          3.89
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        13.1
triple captain      6.5
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.