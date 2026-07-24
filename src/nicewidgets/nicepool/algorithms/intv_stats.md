# interval statistics

in some of our raw csv files we have a column that represents the time of an event. for example, in data/kym_event_report.csv we have a column t_start which is the start time (seconds) for a kym event (an event in general). We have another column t_peak which is the peak (in seconds) of the same event.

in this way, we have a time series of events.

i want to start developing algorithms to calculate the interval (intv) statistics for time series events.

intv statistics for a time series column such as t_start with first need to take the difference between successive events in the timeseries. for example, take t_start and calulate the diff() between successive events. If we have n events, the interval between evens has len n-1.

We want two primary interval statistics.

 - the inter-event-interval (iei), basically diff() of the values in the time series (diff() in values for t_start)
 - the instantaneous frequency (inst_freq) between successive events, e.g. the instantaneous frequency between events. afaik, this would just be 1/iei? please examine and make sure htis is true.

 By convention, our interval statistics like iei and inst_freq will be attached or recorded for each event with-respect-to the previous event. Thus, if we have n events, each with a t_start, the iei and isnt_freq values for event i represents the interval (iei) and inst_freq with respoect to the previous event. With this, if we have n=4 t_start like (.1, .2, .3, .4) the iei and inst_freq values will be padded by missing value (nan) for the first event, the iei would look like (nan, .2-.1, .3-.2, .4-.3)

because the master (original) csv represents measurements across a number of different files (kym tif files), and within each kym tif file, we have important grouping such as roi_id. It does not make sense to calculate interval statistics for all time series event, like t_start, across the entire master csv df.

we can only calculate intv stats when we get down to a grouping of rows within the master df.

for the data/kym_event_report.csv master df, we basically only calculate intv statistics (for t_start as an example) after we have done something like this:

 1. filter/reduce by roi_id (we can not use (none) which yields all roi, we need to specify one roi, like roi_id==1)

 2. filter/reduce by rel_path column. each rel_path gives us a raw tif image that was used. this is the core of our intv analysis, we only want to consider t_start within rows with the same rel_path.
 
 3. filter/reduce by event_type. within one roi_id and one rel_path, we have a number of different types of event. we want the final step for intv stats to focus on just one event type and the interval (Diff) between them as well as 1/iei (the inst freq between event)

 to put another way, time series intv analysis like for t_start only make sense when we reduce (i) by roi_id, then by (ii) rel_path, then by (iii) event_type

To review, in the data/kym_event_report.csv, we only want to calculate interval statistics for time series values like t_start after we reduce the master df down to rows with only:

 1. one roi_id
 2. one rel_path
 3. one event_type


 - filter/reduc by 'event_type' column. this is a special case for the master csv/df from data/kym_event_report.csv. this csv has time series events like t_start for different event types (event_type column). we can start by conceptually using a filter for event type like only events with event_type of 'baseline_rise'. note, we currently have these vent types in the original master csv/df: baseline_rise, baseline_drop, nan_gap, etc


# what to do 1

I said, "i want to start developing algorithms to calculate the interval (intv) statistics for time series events."

once we get to the filtered list of, for example t_start (after step 1, 2, 3) we can calculate the interval statistics for the given time-series stat (e.g. t_start).

 - calculate iei and inst_freq (one value per event, first event is nan)
 - from this list of interval stats, i then want an aggregate of functions such as:

     count, min, max, mean, std, sem, cv

 this can be shown as, for example, a pandas dataframe with some narrative notation

  - filtering by roi_id
  - filtering by rel_path
  - filtering by event_type

The table can have one row for each of iei and inst_freq followed by one column for each of the aggregate stat.

The table should indicate the original column we started with, in this example we started with t_start

# what to do 2

take all the details in this prompy and make a new pure python file (only python, pandas, numpy) called intv_stats.py, put it in the nicewidgets/plot_pool_widget/algorithms/ folder.

# conclude

please examine all my requirements in this prompt.

look to the source code in src/ for any needed hints.

this was a long prompt.

please ask any clarifying questions or needed decisions before proceeding.


---
# answer i gave during cursor ai parsing my requirements:

# answers

1. we should assume that within a filtered (roi_id, rel_path) that all events are ordered. the events are even ordered across even_type. this is important and should be noted as an assumption to the algorithm.

2. this is very important. there are cases where successive time series t_start  do have the same value and we will get iei =0. i opt to keep them (stay closer to all the raw data in the csv master df). we should handle them as you say "Keep them and let inst_freq be inf (and handle inf in count/min/max/mean/std/sem/cv)?".

as with (1) above, this is important and should be noted in the algorithm. maybe add an explicit step showing how we filter these 'bad interval'. in the real-world sequential event should not occure at exactly the same time (e.g. t_start)

3. all three:

 - Text/metadata above the table (e.g. as a separate attribute or string)?
 - Extra columns in the DataFrame (e.g. original_column, roi_id, rel_path, event_type)
 - A dict with something like metadata and table

4. "API: pre-filtered vs. master DataFrame".

Accept the master DataFrame plus roi_id, rel_path, and event_type and perform filtering

5. DataFrame with the usual structure but all NaN for the aggregates?

This is important and we may revise in future steps

6. yes, "For count, min, max, mean, std, sem, cv: should we treat NaN like pandas (drop NaN before computing count, mean, std, etc.),"

# conclude

examine my answers and revise your work. continue with implementation.

---
# one more follow up prompt

1. in the final table, for rel_path. can you remove rel_path from the final table but take it apart by '/' and create columns 'grandparent', 'parent', 'tif_file'.

for example, a rel_path of '14d Saline/20251020/20251020_A100_0013.tif' would yield:

 - grandparent '14d Saline'
 - parent '20251020'
 - tif_file '20251020_A100_0013.tif'

2. at the end of the py script, can you run a batch across all unique rel_path in the original csv/df. for example, the current script does the analysis for one rel_path '14d Saline/20251020/20251020_A100_0013.tif'. i want a final summary of all rel_path in original df (actually, after pre-filtering by roi_id)

3. take the algorithm we have in intv_stats.py and create a new jupyter notebook in notebooks/ intv_stats.ipynb

use the src code in interv_stats.py as the ground truth. as we did before, make the notebook a narrative description of our interval statistics computation. include important notes about assumptions, display all intermediate results as pandas df. intersperse narrative md cell with commented code blocks.

4. once that is done, we will want to examine our new interval statistics algorithm and consider how to incorporate it into our nicegui/plotly gui in plot_pool_widget/ and plot_pool_app/

do not edit @nicewidgets/src/nicewidgets/plot_pool_widget/  or @nicewidgets/src/nicewidgets/plot_pool_app/plot_pool_app.py .

once we make any final revisions to our int_stats algorithm. we will want to start a plan to logically and correctly implement it in our nicegui gui.

# please proceed

ask, do not guess

---
# a very important improvement to the intv stats algorithm

when we calculate the actual intv stats iei and inst_freq. we do so across a number of original y-stat values (like t_start). when we get the iei, some t_start diff() result in iei=0. then, inst_freq goes to inf.

when we start to calculate (in the algoritm) and then report the tables like in "Step 2: Time series and iei, inst_freq", we have a number of iei that go to 0, and inst_freq that go to inf.

in our sample code using rel_path '14d Saline/20251020/20251020_A100_0013.tif' we have something like this:

```
t_start	iei	inst_freq
0	0.010	NaN	NaN
1	0.010	0.000	inf
2	0.010	0.000	inf
3	0.010	0.000	inf
4	0.010	0.000	inf
5	21.551	21.541	0.046423
6	21.551	0.000	inf
```

we need a strategy to handle the iei that are coming up as 0

i am not sure the best strategy to do this? i vote to add an intermediate step after we do "Step 2: Time series and iei, inst_freq"" and before "Step 3: Aggregate stats table"

can we filter these values and remove (or specifically ignore) any iei that resolves to 0?

in this example, we have 6 iei (0, 0, 0, 0, 21.541, 0) with the first one by convention always resolving to nan.

can we do the following in this new step (call it step 2.5 for now)

 - remove iei that are 0, we are assuming they are detection errors as two events can not occur at the same time
 - carry this through the rest of the algorithm. importantly store the 'n_original' as 6 in this example and have a final 'n' (e.g. the count aggregate) after removal of iei=0.

 the aggregate table in "Step 3: Aggregate stats table", could do a count on the now filtered event intervals (and inst freq), after removing iei=0. the count in that table is the count of events we actually aggregated (we did not aggregate events that had iei=0).

 for that table can we have a column for the original total number of events before filtering iei=0 out. A column like my proposed 'n_original', the original number of events considered (before filtering for iei=0)

# please examine my requirements in this prompt

update both our py script in intv_stats.py and then when done with that independently update our jupyter notebook in intv_stats.ipynb

# finally

please ask clarifying questions if needed

ask, do not guess
