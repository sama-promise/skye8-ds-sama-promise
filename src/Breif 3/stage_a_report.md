Missing rent: Of 6,340 listings, [] (%) had no rent recorded. This was not random — 100% of missing rent came from listing_source == "informal" (34.7% of informal listings), while agency and portal had zero missing rent. Because there is no reasonable value to impute for these, they were excluded from the modelling sample.

Rent figure: Mean rent was [] XAF, median was [] XAF, skewness was []. Because skewness is well above 1, a few expensive listings pull the mean upward; the median better reflects a typical listing, so I would quote **[]** to a landlord.

Collinearity: [] and [] had the highest correlation (r = [___]), meaning they both essentially measure unit size. Including both in a regression risks unstable coefficients.

Extreme listings: [___] listings were flagged as statistical outliers (rent or area z-score ≥ 3). These were kept, not removed, since they're needed for influence diagnostics in Stage D.