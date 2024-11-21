from pathlib import Path
import unittest
import os
import numpy as np
import tifffile
import inspect
import h5py
import zarr
import shutil

prefix = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 782.5 --nsino-per-chunk 4'
prefix2 = 'tomocupy recon --file-name data/Downsampled_WB.h5 --reconstruction-type full --nsino-per-chunk 4 --rotation-axis 808 --sample-material Pb '
prefix3 = '--filter-1-auto True --filter-2-auto True --filter-3-auto True --sample-density 11.34 --dezinger 3 '
prefix4 = '--filter-1-density 1.85 --filter-2-density 8.9 --filter-3-density 8.9' 
prefix5 = '--filter-1-density 0.0 --filter-2-density 0.0 --filter-3-density 0.0' 
cmd_dict = {
    # '{prefix} ': 28.307,
    # f'{prefix} --reconstruction-algorithm lprec ': 27.992,
    # f'{prefix} --reconstruction-algorithm linerec ': 28.341,
    # f'{prefix} --dtype float16': 24.186,
    # f'{prefix} --reconstruction-algorithm lprec --dtype float16': 24.050,
    # f'{prefix} --reconstruction-algorithm linerec --dtype float16': 25.543,
    # f'{prefix} --binning 1': 12.286,
    # f'{prefix} --reconstruction-algorithm lprec --binning 1': 12.252,
    # f'{prefix} --reconstruction-algorithm linerec --binning 1': 12.259,
    # f'{prefix} --start-row 3 --end-row 15 --start-proj 200 --end-proj 700': 17.589,
    # f'{prefix} --save-format h5': 28.307,
    # f'{prefix} --nsino-per-chunk 2 --file-type double_fov': 15.552,
    # f'{prefix} --nsino-per-chunk 2 --blocked-views [0.2,1]': 30.790,
    # f'{prefix} --nsino-per-chunk 2 --blocked-views [[0.2,1],[2,3]]': 40.849,
    # f'{prefix} --remove-stripe-method fw': 28.167,
    # f'{prefix} --remove-stripe-method fw --dtype float16': 23.945,
    # f'{prefix} --start-column 200 --end-column 1000': 18.248,
    # f'{prefix} --start-column 200 --end-column 1000 --binning 1': 7.945,
    # f'{prefix} --flat-linear True': 28.308,
    # f'{prefix} --rotation-axis-auto auto --rotation-axis-method sift  --reconstruction-type full' : 28.305,
    # f'{prefix} --rotation-axis-auto auto --rotation-axis-method vo --center-search-step 0.1 --nsino 0.5 --center-search-width 100 --reconstruction-type full' : 28.303,
    # f'{prefix} --remove-stripe-method vo-all ': 27.993,
    # f'{prefix} --bright-ratio 10': 32.631,
    # f'{prefix} --end-column 1535': 28.293,
    # f'{prefix} --end-column 1535 --binning 3': 1.82,    
    # f'{prefix2} {prefix3} {prefix5} --beam-hardening-method standard --calculate-source standard': 3255.912,
    # f'{prefix2} {prefix3} {prefix4} --beam-hardening-method standard': 3248.832,
    # f'{prefix2} {prefix3} {prefix4} --beam-hardening-method standard --calculate-source standard': 3254.634,
    # f'{prefix2} {prefix3} {prefix4} --beam-hardening-method standard --calculate-source standard --e-storage-ring 3.0 --b-storage-ring 0.3': 822.178,    
    f'{prefix}  --save-format zarr': 28.307,  
}

class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


class Tests(unittest.TestCase):

    def test_full_recon(self):
        for cmd in cmd_dict.items():
            if 'beam-hardening' in cmd[0]:
                try:
                    import beamhardening
                except:
                    print('Beamhardening is not installed, skip the test')
                    continue

            shutil.rmtree('data_rec',ignore_errors=True)      
            print(f'TEST {inspect.stack()[0][3]}: {cmd[0]}')
            st = os.system(cmd[0])
            self.assertEqual(st, 0)
            ssum = 0
            try:
                file_name = cmd[0].split("--file-name ")[1].split('.')[0].split('/')[-1]
                data_file = Path('data_rec').joinpath(file_name)
                with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
                    data = fid['exchange/data']
                    ssum = np.sum(np.linalg.norm(data[:], axis=(1, 2)))
            except:
                pass
            for k in range(24):
                file_name = cmd[0].split("--file-name ")[1].split('.')[0].split('/')[-1]
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/{file_name}_rec/recon_{k:05}.tiff'))
                except:
                    pass
            #try:
            import time
            time.sleep(1)
            file_name = cmd[0].split("--file-name ")[1].split('.')[0].split('/')[-1]
            data_file = Path('data_rec').joinpath(file_name)

            # Open the Zarr dataset
            fid = zarr.open('data_rec/test_data_rec.zarr', mode='r')

            # Log dataset shapes
            print(fid[0][:].shape, fid[1][:].shape)

            # Access and copy data
            data = fid[0][:].astype(np.float64).copy()
            print(f"Data shape: {data.shape}")
            print(f"Data sample: {data[:5]}")

            # Normalize data
            data = np.abs(data)

            # Calculate the sum of norms
            ssum = np.sum(np.linalg.norm(data, axis=1))  # Adjust axis if needed
            print(f"Computed ssum: {ssum}")
            print(f"Expected value: {cmd[1]}")

            # Perform the test comparison
            self.assertAlmostEqual(ssum, cmd[1], places=0)

if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
