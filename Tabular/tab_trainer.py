from fastai.tabular import *
import random
import hydra
from omegaconf import DictConfig
import logging



@hydra.main(config_path="./conf/config.yaml")
def tab_trainer(cfg: DictConfig) -> None:
    # A logger for this file
    log = logging.getLogger(__name__)
    log.info('Starting Tabular Trainner with Configs:')
    log.info(cfg.pretty())
    random.seed(42)
    # #############################################################################
    # Read the train and test data
    # #############################################################################
    train_df = pd.read_csv(cfg.dataset.path + 'train.csv.gz')
    test_df = pd.read_csv(cfg.dataset.path + 'test.csv.gz')
    train_df = train_df.drop('id', 1)
    test_df = test_df.drop('id', 1)

    # #############################################################################
    # Preprocess and data bunch
    # #############################################################################
    val_set_size = math.ceil(len(train_df) * .05)
    valid_idx = random.sample(range(0, len(train_df)-1), val_set_size)
    procs = [FillMissing, Normalize]
    dep_var = 'target'
    path = cfg.dataset.path + 'train.csv.gz'
    valid_idx = valid_idx
    data = TabularDataBunch.from_df(
        path=path, df=train_df, procs=procs, dep_var=dep_var, test_df=test_df, valid_idx=valid_idx)

    # #############################################################################
    # Train
    # #############################################################################
    #layers = [512, 256, 128]
    layers = [32, 16]
    learn = tabular_learner(data, layers=layers, metrics=[accuracy, dice])
    learn.fit_one_cycle(1, 1e-2)

    # #############################################################################
    # Test Predictions
    # #############################################################################
    for i in range(18,20):
        log.info('Prediction on training set item <' +  str(i) +  '> actual is: ' +  str(train_df.iloc[i]['target']))
        log.info(learn.predict(train_df.iloc[i]))
        log.info('Prediction on test set item <' + str(i) + '> actual is unknown: ')
        log.info(learn.predict(test_df.iloc[i]))

if __name__ == "__main__":
    tab_trainer()
