import os
import numpy as np
import pandas as pd


def load_selected_data_files_paths(configs):
    data_folders = configs['data']["data_folders"]
    data_paths = get_filespaths(data_folders)
    tickers_of_files = [os.path.split(x)[1].split("_")[0] for x in data_paths]
    meta_data = pd.read_excel(configs["data"]["meta_data_file_path"]).set_index('Symbol')
    available_tickers_metadata = meta_data.index.values

    both_available, ind1, ind2 = np.intersect1d(tickers_of_files, available_tickers_metadata, return_indices=True)

    meta_data = meta_data.loc[both_available]
    data_paths = list(np.array(data_paths)[ind1])

    # --------------------- Filter out some tickers (Only use samples of filters) ----------------------------
    filters = configs["data_selection"]["ticker_filtering"]
    selected_data_paths, selected_meta_data = filter_data(data_paths, meta_data, filters)

    #
    # # --------------------- Finally show number of samples from each action and each class ---------------------
    # c = selected_meta_data_per_key['action']
    # print("Selected number of each action\n", find_number_of_samples_based_on_a_criteria(c))
    print("selected_meta_data.shape = ", selected_meta_data.shape)
    print(selected_meta_data['Sector'].value_counts())
    return selected_data_paths, selected_meta_data


def get_filespaths(data_folders):
    """
    param: data_folders: the main folders that has all data files, either in its root or its sub folders
    return: list of data paths
    """
    data_paths = []
    for data_folder in data_folders:
        for path, _, files in os.walk(data_folder):
            data_paths.extend([(os.path.join(path, name)) for name in files if name.endswith(('.csv', '.pkl'))])
    data_paths.sort()

    return data_paths


def filter_data(data_paths, meta_data, filters):
    """
    Function that filter out (choose) some captures (CSIs) according to a filters (which are defined by the
    JSON config file)
    :param data_paths: list of data absolute paths
    :param meta_data: meta data (type: df)
    :param filters: filters to use for filtering data!
    :return: list of selected data paths and the corresponding meta data
    """

    filtering = []
    criteria = list(filters.keys())
    criteria.remove('MarketCap_in_Billion')

    min_market_cap, max_market_cap = filters['MarketCap_in_Billion']
    market_cap_filter = (meta_data['Market Cap'] > min_market_cap) & (meta_data['Market Cap'] < max_market_cap)
    meta_data = meta_data[market_cap_filter]
    data_paths = np.array(data_paths)[market_cap_filter.values]

    for i, criterion in enumerate(criteria):

        if "All" in filters[criterion] or "All" in filters[criterion][0]:
            this = np.ones(len(meta_data), dtype='bool')
        else:
            this = (np.array(meta_data[criterion]) == filters[criterion][0])
            this_filter = filters[criterion]
            for item in this_filter:
                this = this | (np.array(meta_data[criterion]) == item)
        filtering.append(this)

    filtering = np.array(filtering)
    # Perform element-wise "Logical And"
    filtering = np.sum(filtering, axis=0) == filtering.shape[0]

    selected_data_paths = [data_paths[x] for x in range(len(data_paths)) if filtering[x]]
    selected_meta_data = meta_data[filtering]

    return selected_data_paths, selected_meta_data


def random_selection_specific_amount_of_each_class(csi_paths, meta_data_per_sample):

    return csi_paths, meta_data_per_sample
