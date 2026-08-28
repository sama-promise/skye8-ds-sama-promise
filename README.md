# skye8-ds-sama-promise
Data Science internship RoadMap 
Maintained by Sama Promise Njemchama as part of the Skye8 Data Science internship programme.

#
## Distance benchmark

I compared my two distance functions on the 173 real water points to see how much faster the numpy version actually is compared to a plain python loop.

Loop version (pairwise_distance_loop): 0.3548s for 3 runs
Numpy version (pairwise_distance): 0.0028s for 3 runs

That's about 128x faster with numpy. Makes sense since the loop version is doing 173*173 = ~30,000 individual python-level calculations one at a time, while the numpy version does the whole thing as one array operation.