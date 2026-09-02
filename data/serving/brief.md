# Gameweek 3 brief

**Squad cost** £99.7m  ·  **expected points** 73.8  ·  **captain** Haaland (vice Saka)

## Starting XI
```
  web_name position      team  price  expected_points
  Tzolakis       GK Hull City    4.5             5.36
     Guéhi      DEF  Man City    6.0             6.03
      Egan      DEF Hull City    4.0             5.50
   Gabriel      DEF   Arsenal    8.0             5.17
     Ajayi      DEF Hull City    4.1             5.16
Szoboszlai      MID Liverpool    7.0             6.43
      Saka      MID   Arsenal    9.5             6.00
 M.Sangaré      MID Brentford    5.7             5.61
  Anderson      MID  Man City    6.4             5.33
   Haaland      FWD  Man City   15.5             8.28
      Isak      FWD Liverpool    9.0             6.63
```

## Bench
```
     web_name position      team  price  expected_points
        Dedić      DEF Newcastle    4.5             4.36
Calvert-Lewin      FWD     Leeds    6.0             3.74
        Gomez      MID  Brighton    5.0             4.05
   Verbruggen       GK  Brighton    4.5             3.21
```

## Captaincy
`cost_vs_best` is what you give up by overriding the recommendation.
```
  web_name      team  expected_points  captain_points  cost_vs_best
   Haaland  Man City             8.28           16.56          0.00
      Isak Liverpool             6.63           13.26          3.30
Szoboszlai Liverpool             6.43           12.86          3.70
     Guéhi  Man City             6.03           12.06          4.50
      Saka   Arsenal             6.00           12.00          4.56
```

## Chips
**Hold.** Nothing this week beats what the remaining windows offer.
```
bench boost        15.4
triple captain      8.3
```

## Caveats
- Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as the least reliable component.
- Defensive contributions are modelled on a single season (2025-26) with Poisson counts. They are measurably overdispersed, but a negative binomial improved defenders and worsened midfielders and forwards, so it was not applied.
- The model reads no press conferences. Late team news is yours to apply.