# Data Quality Report

This is a record of every defect I found in the raw data, how many rows it affected, what I did about it, and what someone reusing this table later needs to know.

## 1. Three date formats mixed together
installed_on, inspected_on, reported_on and fixed_on all contained dates in three different formats: YYYY-MM-DD, "D Mon YYYY" (e.g. 14 Jun 2025), and DD/MM/YYYY. I wrote one shared function, parse_flexible_date, that tries all three formats and raises an error if anything doesn't match any of them. All real values in the data matched one of the three formats, so no rows were dropped for this.

## 2. functional column stored inconsistently
The functional column in inspections.csv had at least six different representations of true/false: yes, True, TRUE, 1, no, FALSE, 0. I normalized this into a real boolean by lowercasing and checking against a known set of "true" values. Anything not in that set is treated as false.

## 3. queue_minutes had a unit attached
Some values were plain numbers (e.g. 10), others had " min" attached (e.g. "30 min"). I stripped "min" and whitespace, then converted to numeric. Anything that still couldn't convert becomes missing (NaN) rather than crashing the script.

## 4. depth_m had a unit attached
Same issue as queue_minutes - some values were plain numbers, others had " m" attached (e.g. "50.4 m"). Cleaned the same way.

## 5. cost_xaf had thousands separators
Costs were written like "48,000" with a comma. Left as text, summing this column concatenated the digit strings together into absurd numbers instead of adding them. I stripped commas before converting to numeric.

## 6. Duplicate inspection records
220 rows in inspections.csv were duplicates on the same point_id and inspected_on. There's no way to know for certain which one is "correct," so I made a decision: keep the one with the highest inspection_id (assuming it's the most recently entered or corrected record) and drop the other. This is documented here so it's clear this was a judgment call, not an obviously correct answer.

## 7. Inspections referencing points that don't exist
260 rows in inspections.csv (after deduplication) had a point_id that doesn't appear anywhere in water_points.csv. I dropped these rows rather than keeping them, since there's no way to attach them to a real water point for analysis. If someone wanted to investigate these further (e.g. a data entry error in the point_id), they are not currently retained anywhere - they are simply excluded from the merged table.

## 8. Repairs with no fixed_on date
25 repairs out of 277 have no fixed_on date, meaning the point was never confirmed fixed. This is the most important edge case in the whole dataset - treating these as "fixed" would make the programme's downtime numbers dishonestly low. Instead, I added an is_unresolved flag (True for these 25 rows), and computed downtime_days using the analysis cutoff date (2026-08-25) as a stand-in end date for unresolved repairs, rather than leaving downtime blank or assuming zero downtime. Anyone using downtime_days should be aware that for unresolved repairs, this number represents "days unresolved so far," not a final repair duration.

## Summary for a downstream analyst
- Trust functional, queue_minutes, depth_m, and cost_xaf as cleaned numeric/boolean columns - do not re-derive them from the raw CSVs without applying the same cleaning steps.
- The merged analysis table only includes inspections that reference a real, known water point.
- is_unresolved and downtime_days on the repairs table must be checked together - a large downtime_days value could mean either a long repair or a repair that is still ongoing.
- Duplicate inspections were resolved by keeping the highest inspection_id per point/date pair. If the source system's ID numbering doesn't reliably indicate recency, this assumption should be revisited.