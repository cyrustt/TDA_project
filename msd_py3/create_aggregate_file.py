# create_aggregate_file.py  (Python 3)
import os, sys, time, datetime
import utils
import hdf5_utils as HDF5

def die_with_usage():
    print('create_aggregate_file.py')
    print('   by T. Bertin-Mahieux (2011) Columbia University')
    print('   tb2332@columbia.edu')
    print('')
    print('Creates an aggregate file from all song file (h5 files)')
    print('in a given directory.')
    print('Aggregate files contain many songs; arrays are stored once and indexed.')
    print('')
    print('usage:')
    print('   python create_aggregate_file.py <H5 DIR> <OUTPUT.h5>')
    print('PARAMS')
    print('   H5 DIR     - directory containing .h5 files (recursively)')
    print('   OUTPUT.h5  - filename of the aggregate file to create')
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        die_with_usage()

    maindir = sys.argv[1]
    output  = sys.argv[2]

    if not os.path.isdir(maindir):
        print(f'ERROR: directory "{maindir}" does not exist.')
        sys.exit(1)
    if os.path.isfile(output):
        print(f'ERROR: file "{output}" exists; delete or choose a new filename.')
        sys.exit(1)

    t0 = time.time()
    allh5 = utils.get_all_files(maindir, ext='.h5')
    print(f'Found {len(allh5)} H5 files.')

    HDF5.create_aggregate_file(output, expectedrows=len(allh5), summaryfile=False)
    print('Aggregate file created; filling…')

    h5 = HDF5.open_h5_file_append(output)
    HDF5.fill_hdf5_aggregate_file(h5, allh5, summaryfile=False)
    h5.close()

    dt = str(datetime.timedelta(seconds=time.time() - t0))
    print(f'Aggregated {len(allh5)} files in: {dt}')
