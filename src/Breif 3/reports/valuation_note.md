# Rent Valuation Note

## What drives rent in Douala and Yaounde

Location is by far the biggest factor. Median rents range from about
280,000 XAF in Bastos down to 109,000 XAF in PK14 — nearly a 3x gap
between neighbourhoods, holding the size of the flat roughly constant.
Douala rents run about 18,000 XAF higher than Yaounde on average.

Unit size matters clearly: each additional square metre adds roughly
1% to the rent, and each extra bedroom adds a further meaningful
increase, holding location and fittings constant. Fittings and
condition also matter — a furnished flat rents for about 16% more
than an unfurnished one, gated security adds about 7%, and a
generator, borehole, or tiled flooring each add smaller but
measurable amounts. Property type (apartment vs. studio vs. villa)
makes almost no difference once location and size are accounted for.

One caution: flats with a generator also happen to rent for more on
average, but generators are heavily concentrated in specific
neighbourhoods — so the apparent "generator effect" is partly a
location effect in disguise, and only partially reflects the
generator itself.

## What the model is worth

The model explains about 86% of the variation in rent (on a log
scale, which is the more reliable way to measure fit for this kind
of skewed data). That headline number looks strong, but the figure
the agency should actually care about is the typical error in real
francs, below — R² alone can be misleading for pricing decisions.

## Typical error in francs

On listings the model had never seen, predictions were off by about
50,000 XAF on average (MAE). A second measure, which penalises large
misses more heavily, comes out at about 122,000 XAF (RMSE) — the gap
between these two numbers tells us that most predictions are
reasonably close, but a smaller number of expensive, larger
properties are missed by a lot. In percentage terms, the typical
prediction is off by about 22% of the actual rent.

## What this model should not be used for

The model was built only on listings where rent was actually
recorded. About 8% of listings — entirely from informally-sourced
listings — had no rent on file, and there is no reliable way to know
what those would have rented for. The model should not be trusted
for informally-sourced flats, since it has effectively no data on
that segment.

The model is also weakest at the high end of the market — large,
expensive properties show the biggest errors, both because there are
fewer of them to learn from and because their prices vary more. A
quote for a premium property should be treated as a rougher starting
point for negotiation, not a precise figure.

## What the model cannot see

The data captures neighbourhood, size, age, and fittings, but not
everything that affects what a flat is actually worth. It cannot see
the physical condition of the road outside, the specific building's
maintenance history, natural light, noise, or a landlord's
willingness to negotiate. It also cannot see anything about a
tenant's own bargaining position. The model's estimate is a starting
point for a conversation, not a replacement for someone viewing the
property.