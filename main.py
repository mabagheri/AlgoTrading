__copyright__ = "MohammadAli"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import time
import json
import random
import numpy as np
import pandas as pd
from core.initialization.load_data_file_paths import load_selected_data_files_paths
from core.feature_extraction.FE_v0 import load_preprocess_extract_features
from core.classification.classify_v2 import classify
from core.classification.HPO import Perform_HPO_GridSearch
import warnings
# import yaml
warnings.filterwarnings("ignore")
np.random.seed(135)
random.seed(135)

configurations = json.load(open('configurations/configs_Canada.json', 'r'))
# configurations = json.load(open('Results/2022.01.28-2325_All_10_3000B_20210501_to_20211231, 7_in_14days/configs.json', 'r'))


# with open('configurations/configs_test.yaml', 'w') as yml:
#     yaml.dump(configurations, yml, allow_unicode=True)
# import yaml, json, sys
# sys.stdout.write(yaml.dump(json.load(sys.stdin)))


def main(configs):
    # configs['saving_folder'] = generate_saving_folder(configs)
    start_time = time.time()

    # 1:
    data_paths, metadata = load_selected_data_files_paths(configs)

    # 2:
    if configs['feature_extraction'].get('Load pre-generated features', False):
        sfn = get_XY_saving_path(configs)
        print(f"Load pre-generated features: {sfn}")
        Xy = pd.read_csv(sfn, parse_dates=['Date'])
    else:
        Xy = load_preprocess_extract_features(data_paths, metadata, configs)
    print("Xy.shape", Xy.shape)
    u, c = np.unique(Xy.columns, return_counts=True)
    print(u[c > 1], c[c > 1])

    # 3:
    if configs['classification'].get('perform_HPO_GridSearch', False):
        configs = Perform_HPO_GridSearch(configs, Xy)

    classification_results = classify(Xy, configs)
    print(classification_results)

    print("Total elapsed time (seconds) : ", time.time() - start_time)

    return classification_results


def get_XY_saving_path(configs):
    sector = ' '.join(configs['data_selection']['ticker_filtering']['Sector'])
    m_caps = (configs['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
    days = [x.replace('-', '') for x in configs['data_selection']['time_filtering']]
    sfn = f"GeneratedDatasets/{sector}_{days[0]}_to_{days[1]}_MarketCap={m_caps[0]}to{m_caps[1]}B.csv"
    return sfn


def generate_saving_folder(configs):
    market = configs['data_selection']['ticker_filtering']['Country'][0]
    sector = ' '.join(configs['data_selection']['ticker_filtering']['Sector'])
    mc = (configs['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
    days = [x.replace('-', '') for x in configs['data_selection']['time_filtering']]
    period = configs['Target_definition']["Period"]
    desired_increase = configs['Target_definition']["Increase_pct"]
    saving_folder = f"Results/{time.strftime('%Y.%m.%d-%H%M')}_{market}_{sector}_{mc[0]}_{mc[1]}B_{days[0]}_to_{days[1]}, " \
                    f"{desired_increase}_in_{period}days/"
    if not os.path.exists(saving_folder):
        os.mkdir(saving_folder)

    return saving_folder


if __name__ == '__main__':
    # try:
    configurations['saving_folder'] = generate_saving_folder(configurations)
    main(configurations)

    from distutils.dir_util import copy_tree
    from shutil import copyfile

    os.mkdir(f"{configurations['saving_folder']}/Codes")
    copy_tree("core", f"{configurations['saving_folder']}/Codes/core")
    copyfile('main.py', f"{configurations['saving_folder']}/Codes/main.py")

    # except:
    #     print(" ***************** Error **************** ")
    #     import shutil
    #     shutil.rmtree(configurations['saving_folder'])
