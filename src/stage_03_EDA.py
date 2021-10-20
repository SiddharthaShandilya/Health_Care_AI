import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os
from src.utils.all_utils import create_directory, read_yaml, save_local_df
from sklearn.ensemble import ExtraTreesRegressor

def eda(config_path):

    config = read_yaml(config_path)

    artifacts_dir = config["artifacts"]['artifacts_dir'] # points to artifacts folder
    data_local_dir = config["artifacts"]['data_local_dir'] # points to data folder inside artifacts
    real_data_dir = config["artifacts"]['real_data_dir'] # points to Real-Data folder inside artifacts/data
    real_data_combined_file = config["artifacts"]['real_data_combined_file'] # points to  Real_Combine.csv 
    cleaned_real_data_file = config["artifacts"]['cleaned_real_data_file'] # points to  cleaned_real_data_file.csv 

    reports_dir = config["artifacts"]['reports']['reports_dir']
    graphs_dir = config["artifacts"]['reports']['graphs']['graphs_dir']
    graphs_dir_path = os.path.join(artifacts_dir,reports_dir, graphs_dir)


    real_data_combined_file_path = os.path.join(artifacts_dir, data_local_dir, real_data_dir, real_data_combined_file)

    cleaned_real_data_file_path = os.path.join(artifacts_dir, data_local_dir, real_data_dir, cleaned_real_data_file)

    #data = eda(real_data_combined_file_path)
    print(" real_data_combined_file_path = {}".format(real_data_combined_file_path))
    
    df = pd.read_csv(real_data_combined_file_path) 
    print(" FILE LOCATION : {}".format(real_data_combined_file_path))
    

    df.dropna(inplace= True)
    X = df.iloc[:,:-1] # removing the last column
    y = df.iloc[:,-1]

    # finding co- relation
    corr_mat = df.corr()
    top_corr_features = corr_mat.index
    print("top_corr_features = {}".format(top_corr_features))

    
    #-------------------------------------------------------------------------------
    #                          feature importance  
    #-------------------------------------------------------------------------------

    model = ExtraTreesRegressor()
    model.fit(X,y)
    print("model.feature_importances_ = {}".format(model.feature_importances_))

    #-------------------------------------------------------------------------------
    #                           UPLOADING GRAPHS   
    #-------------------------------------------------------------------------------
    if not os.path.exists("{}".format(graphs_dir_path)):
        create_directory([os.path.join(os.path.join(artifacts_dir, reports_dir, graphs_dir))])

    fig = plt.figure()
    feat_importance = pd.Series(model.feature_importances_, index=X.columns)
    #feat_importance.nlargest(5).plot(kind='barh')
    feat_importance.nlargest(8).plot(kind='barh')
    fig.savefig('{}/feat_importance.png'.format(graphs_dir_path))
    print(" # "*20 + " feature_importance graph created")

    #-------------------------------------------------------------------------------
    # SNS_PAIRPLOT
    #-------------------------------------------------------------------------------
    fig = plt.figure()
    sns.pairplot(df)
    fig.savefig('{}/sns_pairplot.png'.format(graphs_dir_path))
    print(" # "*20 + " SNS_PAIRPLOT GRAPH created")

    #-------------------------------------------------------------------------------
    # SNS HEAT_MAP 
    #-------------------------------------------------------------------------------
    fig = plt.figure()
    corr_mat = df.corr()
    top_corr_features = corr_mat.index
    plt.figure(figsize=(20,20))
    g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="YlGnBu")
    fig.savefig('{}/sns_heatmap.png'.format(graphs_dir_path))
    print(" # "*20 + " SNS HEAT_MAP  GRAPH created")

    #-------------------------------------------------------------------------------
    # SNS_DISTPLOT
    #-------------------------------------------------------------------------------
    
    fig = plt.figure()
    sns.distplot(y)
    fig.savefig('{}/SNS_DISTPLOT(Y).png'.format(graphs_dir_path))
    print(" # "*20 + "SNS_DISTPLOT(Y).png   created")

    #return df 
    #-------------------------------------------------------------------------------
    # SAVING THE DATA TO real_data_combined_file_path
    #-------------------------------------------------------------------------------
    print("----"*30)
    save_local_df(df, cleaned_real_data_file_path)
    print(" cleaned_real_data_file_path = {}".format(cleaned_real_data_file_path))
    print("----"*30)


             




if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    config_path=parsed_args.config
    
    eda(config_path=parsed_args.config)