from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
def split_data(config_path, params_path):

    config = read_yaml(config_path)
    params = read_yaml(params_path)


    artifacts_dir = config["artifacts"]['artifacts_dir'] # points to artifacts folder
    data_local_dir = config["artifacts"]['data_local_dir'] # points to data folder inside artifacts
    real_data_dir = config["artifacts"]['real_data_dir'] # points to Real-Data folder inside artifacts/data
    split_data_dir = config["artifacts"]["split_data"]['split_data_dir'] # points to split_data folder inside artifacts/data
    train_data_dir = config["artifacts"]["split_data"]['train_data_dir'] # points to split_data folder inside artifacts/data
    test_data_dir = config["artifacts"]["split_data"]['test_data_dir'] # points to split_data folder inside artifacts/data

    real_data_combined_file = config["artifacts"]['cleaned_real_data_file'] # points to  Real_Combine.csv 
    train_data_file = config["artifacts"]["split_data"]['train_data_file'] 
    test_data_file = config["artifacts"]["split_data"]['test_data_file'] 

    real_data_dir_path = os.path.join(artifacts_dir, data_local_dir, real_data_dir)
    train_data_dir_path = os.path.join(artifacts_dir, split_data_dir, train_data_dir)
    test_data_dir_path = os.path.join(artifacts_dir, split_data_dir, test_data_dir)

    real_data_combined_file_path = os.path.join(real_data_dir_path, real_data_combined_file)
    train_data_file_path = os.path.join(train_data_dir_path, train_data_file)
    test_data_file_path = os.path.join(test_data_dir_path, test_data_file)

    print("#"*20 + " real_data_combined_file_path " + "#"*20)
    print(" real_data_dir_path = {}".format(real_data_dir_path))
    print(" real_data_combined_file_path = {}".format(real_data_combined_file_path))

    print(" train_data_file_path = {}".format(train_data_file_path))
    print(" test_data_file_path = {}".format(test_data_file_path))
    print("#"*20 + " real_data_combined_file_path " + "#"*20)

    if not os.path.exists("{}".format(train_data_file_path)):                           
        create_directory([os.path.join(artifacts_dir, split_data_dir, train_data_dir)])

    if not os.path.exists("{}".format(test_data_file_path)):                                
        create_directory([os.path.join(artifacts_dir, split_data_dir, test_data_dir)])

    
    # fetching parameters
    split_ratio = params["split_data"]["test_size"]
    random_state = params["split_data"]["random_state"]

    df = pd.read_csv(real_data_combined_file_path)

    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)

    split_data_dir = config["artifacts"]["split_data"]["split_data_dir"]

    create_directory([os.path.join(artifacts_dir, split_data_dir)])



    
    for data, data_path in (train, train_data_file_path), (test, test_data_file_path):
        save_local_df(data, data_path)
        #print( " {} created".format(data_path))

    


if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    config_path=parsed_args.config
    
    split_data(config_path=parsed_args.config, params_path=parsed_args.params)
    