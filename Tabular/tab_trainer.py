from fastai.tabular import *
import random
import hydra
from omegaconf import DictConfig
import logging
from fastai.callbacks.oversampling import OverSamplingCallback
import torch;
from tqdm import tqdm

@hydra.main(config_path="./conf/config.yaml")
def tab_trainer(cfg: DictConfig) -> None:
    # A logger for this file
    log = logging.getLogger(__name__)
    log.info('Starting Tabular Trainner with Configs:')
    log.info('Torch Cuda is: ' + str(torch.cuda.is_available()))
    log.info(cfg.pretty())

    # #############################################################################
    # For Reproducability
    # #############################################################################
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # #############################################################################
    # Read the train and test data
    # #############################################################################
    train_df = pd.read_csv(cfg.dataset.training_data)
    test_df = pd.read_csv(cfg.dataset.testing_data)
    train_df = train_df.drop('id', 1)
    test_df = test_df.drop('id', 1)

    # #############################################################################
    # Preprocess and data bunch
    # #############################################################################
    val_set_size = math.ceil(len(train_df) * .05)
    valid_idx = random.sample(range(0, len(train_df)-1), val_set_size)
    procs = [FillMissing, Normalize]
    dep_var = 'target'
    path = cfg.dataset.training_data
    valid_idx = valid_idx
    data = TabularDataBunch.from_df(
        path=path, df=train_df, procs=procs, dep_var=dep_var, test_df=test_df, valid_idx=valid_idx)

    # #############################################################################
    # Train
    # #############################################################################
    layers = [2048, 1024]
    learn = tabular_learner(data, layers=layers, metrics=[accuracy, dice], callback_fns=[OverSamplingCallback], path=cfg.dataset.model_data)

    # Determine optimal lr - do once - found lr = 1e-4
    # learn.lr_find()
    # fit = learn.recorder.plot(return_fig=True)
    # fit.savefig(cfg.dataset.model_data + 'lr_rate.png')
    # log.info('Learning Rate Plot saved to ' + cfg.dataset.model_data + 'lr_rate.png')

    learn.fit_one_cycle(4, 1e-4)
    learn.fit_one_cycle(2, 2e-7, wd=0.3)
    learn.fit_one_cycle(1, 2e-11, wd=0.5)

    # #############################################################################
    # Test Predictions
    # #############################################################################
    for i in range(18,20):
        log.info('Prediction on training set item <' +  str(i) +  '> actual is: ' +  str(train_df.iloc[i]['target']))
        log.info(learn.predict(train_df.iloc[i]))
        log.info('Prediction on test set item <' + str(i) + '> actual is unknown: ')
        log.info(learn.predict(test_df.iloc[i]))

    # #############################################################################
    # Test Predictions
    # #############################################################################
    log.info('Collecting Predictions')
    data = []
    for i in tqdm(range(0,len(test_df)-1)):
        data.append( [i,learn.predict(test_df.iloc[i])[2][1].item()] )

    log.info('Writing Predictions to ' + cfg.dataset.prediction_data)
    out_df = pd.DataFrame(data, columns = ['id', 'target'])
    log.info(out_df.head())
    out_df.to_csv(cfg.dataset.prediction_data, columns = ['id', 'target'], index=False)

    log.info('Run Complete')
    logging.shutdown()

if __name__ == "__main__":
    tab_trainer()
