import sys, os
import numpy as np
from SQLiteConnection import engine, Session
from DBClasses import DBFlare

folder = '/Users/julian/Documents/phd/solar_flares/data/xrs_goes_summaries'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
dontuse_kw = ['dontuse', 'input']
files = [f for f in files if (not any([bad in f for bad in dontuse_kw])) and f.endswith('.txt')]
years = np.sort([(f.removeprefix('goes-xrs-report_')).removesuffix('.txt') for f in files])
date_of_first_in_swpc = np.datetime64('2015-06-29T00:00')

# don't read in 2016/2017, as using SWPC data for that period
years = years[:-2]

names = ['date', 'start', 'end', 'peak', 'pos', 'class', 'sat_num', 'flux', 'region', 'misc1', 'misc2', 'misc3']
# header:       date   start end   peak  pos   class  sat_num flux   region ??     ??      ??
dtypes =       ['U11', 'U4', 'U4', 'U4', 'U8', 'U4', 'U10',   float, int,   float, float,  float]
fixed_widths = [13,     5,    5,    5,    31,   4,    9,      8,     6,     9,     8,      8]

# takes dirty 'date' and times from GOES catalog list and cleans/checks
# returns np.datetime64 objects for start, end, peak
def times_str_clean(date, start, end, peak, year):
    ymd = date
    try:
        assert ymd.startswith(year[-2:])
    except AssertionError:
        if len(ymd) == 11:
            ymd = ymd[5:]
        try:
            assert ymd.startswith(year[-2:])
        except AssertionError:
            print(date, year)

    y = ymd[:2]
    y = '19' + y if int(y[0]) > 6 else '20' + y
    m = ymd[2:4]
    d = ymd[4:]
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
            start_time = np.datetime64(full_start)
        else:
            print(f'Something gone wrong with converting flare {date} {start}')
            return None, None, None
    try:
        end_time = np.datetime64(full_end) # time combining date and end
    except ValueError:
        ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
        if end[:2] in ['24', '25', '26', '27', '28']:
            new_hour = int(end[:2]) - 24
            end = f'{new_hour:02}' + ':' + end[-2:]
            full_end = ymd_plusone + 'T' + end
        # correcting typo in flare starting 1995-03-16 14:13
        elif int(date) == 31777950316 and start == '14:13':
            end = '15:06'
            full_end = ymd + 'T' + end
        else:
            print(f'Something gone wrong with converting flare {date} {start} {end}')
            return None, None, None
        end_time = np.datetime64(full_end)
    try:
        assert end_time >= start_time
    except AssertionError:
        ymd_plusone = str(np.datetime64(ymd) + np.timedelta64(1, "D"))
        if end[:2] in ['00', '01', '02', '03']:
            full_end = ymd_plusone + 'T' + end
            end_time = np.datetime64(full_end)
        # correcting date for unusually long event starting 1998-01-02T23:35
        elif end == '06:30' and start == '23:35':
            full_end = ymd_plusone + 'T' + end
            end_time = np.datetime64(full_end)
        # correcting date for unusually long event starting 2002-03-11T04:04
        elif end == '04:04' and start == '22:48':
            full_end = ymd_plusone + 'T' + end
            end_time = np.datetime64(full_end)
        # correcting date for unusually long event starting 2010-04-17T04:16
        elif end == '04:16' and start == '21:34':
            full_end = ymd_plusone + 'T' + end
            end_time = np.datetime64(full_end)
        else:
            print(f'Check why {end_time} < {start_time}, not including in DB')
            return None, None, None
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
            print(f'Something gone wrong with converting flare {date} {start} {end} {peak}')
            return None, None, None

    return start_time, end_time, peak_time

# converts fclass from a string of Y AB to appropriate peak flux in Wm-2
def fclass_to_peakflux(fclass_str):
    letter = fclass_str[0]
    if letter == ' ':
        return 0
    mult = fclass_str[1:].strip(' ')
    if len(mult)>=2:
        multiplier = float(mult[:-1] + '.' + mult[-1])
    else:
        try:
            multiplier = float(mult)
        except ValueError:
            if mult == '':
                multiplier = 1
            else:
                print(f'Something up with this fclass: {fclass_str}')
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

session = Session(autoflush=False)

for year in years:
    print(f'Adding flares from {year.rstrip("-ytd")} to the database')
    file_path = os.path.join(folder, 'goes-xrs-report_' + year + '.txt')

    with open(file_path) as f:
        data = np.genfromtxt(f, dtype=dtypes, delimiter=fixed_widths, names=names)

    for datum in data:
        start_time, end_time, peak_time = times_str_clean(datum['date'], datum['start'], datum['end'], datum['peak'], year.rstrip('-ytd'))
        if start_time > date_of_first_in_swpc:
            continue
        
        flare_dic = {}
        flare_dic['start_time'] = str(start_time)
        flare_dic['end_time'] = str(end_time)
        if peak_time is not None:
            flare_dic['peak_time'] = str(peak_time)
        else:
            flare_dic['peak_time'] = 'NaT'
        if len(datum['pos'].strip(' ')) > 0:
            flare_dic['location'] = datum['pos']
        else:
            flare_dic['location'] = 'not_recorded'
        flare_dic['fclass'] = datum['class']
        if datum['flux'] > 0:
            flare_dic['integ_flux'] = datum['flux']
        else:
            flare_dic['integ_flux'] = -1
        pflux = fclass_to_peakflux(datum['class'])
        if pflux > 0:
            flare_dic['peak_flux'] = pflux
        else:
            flare_dic['peak_flux'] = -1
        
        region_num = datum['region']
        # correcting typos in region numbers
        if region_num == 1000 and start_time == np.datetime64('1978-05-30T06:19'):
            region_num = 1134
        if region_num == 3121 and start_time == np.datetime64('1981-07-18T11:46'):
            region_num = 3221
        if region_num == 366 and start_time == np.datetime64('1981-08-22T06:58'):
            region_num = 3266
        if region_num == 2102 and (start_time == np.datetime64('1983-03-01T18:24') or start_time == np.datetime64('1983-03-01T18:54')):
            region_num = 4102
        if region_num == 4135 and start_time == np.datetime64('1983-07-04T06:09'):
            region_num = 4235
        if region_num == 4236 and start_time == np.datetime64('1983-07-29T03:53'):
            region_num = 4263
        if region_num == 7500 and start_time == np.datetime64('1993-09-27T01:35'):
            region_num = 7590
        if region_num == 9125 and start_time == np.datetime64('2000-11-09T21:13'):
            region_num = -1
        if region_num == 1 and start_time == np.datetime64('2002-06-14T20:18'):
            region_num = 10001
        if region_num == 422 and start_time == np.datetime64('2003-07-31T07:59'):
            region_num = 10422
        if region_num == 10000 and start_time == np.datetime64('2003-12-21T04:10'):
            region_num = -1
        if region_num == 11281 and start_time == np.datetime64('2011-12-22T13:04'):
            region_num = 11381
        # replace region number for initial 8 events in region 10198 as it leaves east limb then reappears 16d later
        # if these initial flares kept in region, waiting times/correlations would be biased
        if region_num == 10198 and start_time < np.datetime64('2002-11-17T00:00'):
            region_num = -1
        if region_num not in [-1, 0, 1, 2, 4]:
            flare_dic['region_id'] = region_num
        else:
            flare_dic['region_id'] = -1

        session.execute(DBFlare.__table__.insert().prefix_with('OR IGNORE').values(flare_dic))

        session.flush()

session.commit()
engine.dispose()

