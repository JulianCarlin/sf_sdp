import sys, os
import numpy as np
from numpy.lib import recfunctions as rfn
from SQLiteConnection import engine, Session
from DBClasses import DBFlare #, DBRegion
from funcs import progress_bar, time_to_timestring
from time import time

# get list of filenames of daily SWPC flare reports
folder = '/Users/julian/Documents/phd/solar_flares/data/events_SWPC_2015onwards'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
bad = ['events', 'yesterday', 'README']
files = np.sort([f for f in files if not any([f.startswith(b) for b in bad])])

# defining names, dtypes, widths for data read-in
names = ['Event', 'Begin', 'Max', 'End', 'Obs', 'Q', 'Type', 'Loc/Frq', 'Particulars', 'Reg#']
dtypes = ['U4', 'U4', 'U4', 'U4', 'U3', 'U1', 'U3', 'U7', 'U15', 'U4']
fixed_widths = [11, 7, 10, 6, 5, 4, 5, 10, 18, 4]
dtype_name_map = [(n, dt) for n, dt in zip(names, dtypes)]
dtype_name_map.insert(0, ('date', 'U8')) 

# converts fclass from a string of 'Y.AB' to appropriate peak flux in Wm-2
def fclass_to_peakflux(fclass_str):
    letter = fclass_str[0]
    if letter == ' ':
        return 0
    mult = fclass_str[1:].strip(' ')
    try:
        multiplier = float(mult)
    except ValueError:
        if mult == '':
            multiplier = 1
        else:
            print(f'Something up with this fclass: {fclass_str}')
            return 0
    except:
        print(f'Uh oh, uncaptured error for this fclass: {fclass_str}')
        return 0
    possible_letters = ['A', 'B', 'C', 'M', 'X']
    bases = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    try:
        assert letter in possible_letters
    except AssertionError:
        print(f'Tried to convert {fclass_str} into peak-flux, but didn\'t understand first letter')
    try:
        assert multiplier < 10 or letter=='X'
    except AssertionError:
        print(f'Multiplier for flare class ({multiplier}) not less than 10 for non X-class flare')

    base = bases[possible_letters.index(letter)]
    peakflux = base * multiplier
    return peakflux

# takes YYYYMMDD date, start/end/peak times (format HHMM), and cleans + converts to np.datetime64 objects
def times_str_clean(date, start, end, peak):
    y = date[:4]
    m = date[4:6]
    d = date[6:]
    start = start[:2] + ':' + start[2:]
    end = end[:2] + ':' + end[2:]
    peak = peak[:2] + ':' + peak[2:]
    ymd = y + '-' + m + '-' + d
    full_start = ymd + 'T' + start
    full_end = ymd + 'T' + end
    full_peak = ymd + 'T' + peak

    try:
        start_time = np.datetime64(full_start) # time combining date and start
    except ValueError:
        if start.startswith('24'):
            ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
            start = '00:' + start[-2:]
            full_start = ymd_plusone + 'T' + start
        # correcting typo in flare 2020-04-09 00:91
        elif start == '00:91':
            start = '09:10'
            full_start = ymd + 'T' + start
        # correcting typo in flare 2020-05-27 01:84
        elif start == '01:84':
            start = '18:04'
            full_start = ymd + 'T' + start
        else:
            print(f'Something up with {date} (starttime) {start}... Not adding to DB')
            return None, None, None
        start_time = np.datetime64(full_start)
    try:
        end_time = np.datetime64(full_end) # time combining date and end
    except ValueError:
        ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
        if end[:2] in ['24', '25', '26', '27', '28']:
            new_hour = int(end[:2]) - 24
            end = f'{new_hour:02}' + ':' + end[-2:]
            full_end = ymd_plusone + 'T' + end
        # correcting typo in flare 2020-05-29 06:49
        elif end == '00:75':
            end = '07:05'
            full_end = ymd + 'T' + end
        else:
            print(f'Something up with {date} (endtime) {end}... Not adding to DB')
            return None, None, None
        end_time = np.datetime64(full_end)
    try:
        assert end_time >= start_time
    except AssertionError:
        if ((start.startswith('23') or start.startswith('22') or start.startswith('21'))
            and (end.startswith('00') or end.startswith('01'))):
            ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
            full_end = ymd_plusone + 'T' + end
        # correcting typo in flare 2020-01-10 11:52
        elif end == '01:24' and start == '11:52':
            end = '12:04'
            full_end = ymd + 'T' + end
        # correcting typo in flare 2020-04-07 01:46
        elif end == '00:26' and start == '01:46':
            end = '02:06'
            full_end = ymd + 'T' + end
        # correcting typo in flare 2020-05-28 09:31
        elif end == '01:03' and start == '09:31':
            end = '10:03'
            full_end = ymd + 'T' + end
        # correcting typo in flare 2020-05-29 03:39
        elif end == '00:40' and start == '03:39':
            end = '04:00'
            full_end = ymd + 'T' + end
        # flare at 2022-01-09 15:52 seems misplaced, not including in DB
        elif end == '05:31' and start == '15:52':
            return None, None, None
        # correcting typo in flare 2022-04-27 14:14
        elif end == '01:27' and start == '14:14':
            end = '15:27'
            full_end = ymd + 'T' + end
        else:
            print(f'Check why {end_time} < {start_time}, not including in DB')
            return None, None, None
        end_time = np.datetime64(full_end)
    try:
        peak_time = np.datetime64(full_peak) # time combining date and peak
    except ValueError:
        if peak[:2] in ['24', '25', '26', '27']:
            new_hour = int(peak[:2]) - 24
            peak = f'{new_hour:02}' + ':' + peak[-2:]
            ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
            full_peak = ymd_plusone + 'T' + peak
            peak_time = np.datetime64(full_peak)
        elif peak == '  :  ' or peak == '//://':
            peak_time = None
        else:
            print(f'Something up with {date} (peaktime) {peak}, not including in DB')
            return None, None, None

    return start_time, end_time, peak_time

session = Session(autoflush=False)

# load all daily files into one big np structured array
first_file = True
for i, f in enumerate(files):
    file_path = os.path.join(folder, f)
    with open(file_path) as opened:
        data = np.genfromtxt(opened, dtype=dtypes, delimiter=fixed_widths, names=names, missing_values='////', skip_header=12, comments='\n')
    date = f.rstrip('events.txt')
    try:
        dates_arr = np.full(data.shape[0], date, dtype=[('date', 'U8')])
    except IndexError:
        dates_arr = np.full(1, date, dtype=[('date', 'U8')])
    full_data = rfn.merge_arrays((dates_arr, data), flatten=True)

    if not first_file:
        total_data = np.hstack((total_data, full_data))
    else:
        total_data = full_data
        first_file = False

# separate flares seen by GOES and all others (others used for location data)
goes_mask = np.array([datum['Obs'].startswith('G') for datum in total_data])
goes_data = np.array(total_data[goes_mask], dtype=dtype_name_map)
not_goes_data = np.array(total_data[~goes_mask], dtype=dtype_name_map)

init = time()
cumul_time = 0
num_goes = np.sum(goes_mask)
for i, datum in enumerate(goes_data):
    progress_bar(i, num_goes, cumul_time)
    t = time()

    flare_dic = {}
    
    start_time, end_time, peak_time = times_str_clean(datum['date'], datum['Begin'], datum['End'], datum['Max'])
    flare_dic['start_time'] = str(start_time)
    flare_dic['end_time'] = str(end_time)
    # sometimes peak_time is missing, but that's okay
    if peak_time is not None:
        flare_dic['peak_time'] = str(peak_time)
    else:
        flare_dic['peak_time'] = 'NaT'
    # select event(s) that share 'Event' marker and region, for filling in location data of flare
    same_event_diff_obs = [other for other in not_goes_data if (other['Event']==datum['Event'] and other['Reg#']==datum['Reg#'])]
    # only use flares that are confirmed same region number
    if len(same_event_diff_obs) > 0 and datum['Reg#'].strip(' ') != '':
        pos = [event['Loc/Frq'] for event in same_event_diff_obs if event['Loc/Frq'].startswith('N') or event['Loc/Frq'].startswith('S')]
        if len(pos) == 1:
            position = pos[0].strip(' ')
            flare_dic['location'] = position
    else:
        flare_dic['location'] = 'not_recorded'
    
    # integrated flux in last 7 digits of 'Particulars' column for GOES events
    try:
        flux = float(datum['Particulars'][-7:])
        flare_dic['integ_flux'] = flux
        # flare.integ_flux = flux
    except ValueError:
        print(f"Could't convert '{datum['Particulars'][-7:]}' to a float")
    # fclass in the first 6 digits
    flare_dic['fclass'] = datum['Particulars'][:6].strip(' ')
    pflux = fclass_to_peakflux(flare_dic['fclass'])
    if pflux > 0:
        flare_dic['peak_flux'] = pflux
    else:
        flare_dic['peak_flux'] = -1

    # adding 10k to region number to align with numbering from pre-2015
    region_num =  int('1' + datum['Reg#'])
    # don't add region numbers if in the "missing" list
    if region_num not in [-1, 0, 1, 2, 4]:
        if region_num == 12984 and start_time == np.datetime64('2021-11-06T22:01'):
            region_num = 12894
        if region_num == 12655 and start_time == np.datetime64('2017-07-11T01:09'):
            region_num = 12665
        if region_num == 12655 and start_time == np.datetime64('2017-07-16T10:25'):
            region_num = 12665
        if region_num == 12680 and start_time == np.datetime64('2021-09-01T03:03'):
            region_num = 12860
        if region_num == 12680 and start_time == np.datetime64('2021-09-01T04:27'):
            region_num = 12860
        if region_num == 12282 and start_time == np.datetime64('2021-05-10T23:46'):
            region_num = 12822
        if region_num == 12807 and start_time == np.datetime64('2021-12-18T11:17'):
            region_num = 12907
        if region_num == 12807 and start_time == np.datetime64('2021-12-18T17:27'):
            region_num = 12907

        flare_dic['region_id'] = region_num
    else:
        flare_dic['region_id'] = -1

    # insert or ignore SQL will not add row to table if joint unique condition not met
    # unique condition: start_time, end_time, location, fclass, region_id not all matching
    session.execute(DBFlare.__table__.insert().prefix_with('OR IGNORE').values(flare_dic))
    session.flush()
    cumul_time += time() - t

print(f'\nTotal time to load {len(goes_data)} events into database: {(time() - init)/60:.2f}min')
session.commit()

engine.dispose()
