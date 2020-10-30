from dask.distributed import Client, LocalCluster, wait, progress
from dask_jobqueue import SLURMCluster
import yaml
import os

from .helpers import rec_dd
from .helpers import scan_dict
from .quantileRegression_chain import quantileRegression_chain as QRC
from .quantileRegression_chain_disc import quantileRegression_chain_disc as QRCd

import logging
logger = logging.getLogger(__name__)



class QRCScheduler:
    """Set up a scheduler for quantile chain regression. Based on the config file
    it is fed with, it creates a variable amount of QRC/QRCd objects depending on
    the combinations between the type of datasets passed (data/mc), detectors
    (EB/EE) and variables (SS/iso-ph/iso-ch)

    Args:
        config_file_name (str): name of a yaml file used to set up the quantile
        regressors

        Example:

        dataframes:
          data:
            EB:
              SS:
                df_data_EB_train.h5
              iso:
                df_data_EB_Iso_train.h5
            EE:
              SS:
                df_data_EE_train.h5
              iso:
                df_data_EE_Iso_train.h5
          mc:
            EB:
              SS:
                df_mc_EB_train.h5
              iso:
                df_mc_EB_Iso_train.h5
            EE:
              SS:
                df_mc_EE_train.h5
              iso:
                df_mc_EE_Iso_train.h5

        year:
          2017

        datasets:
          ['data', 'mc']

        detectors:
          ['EB', 'EE']

        variables:
          SS:
            ['probeCovarianceIeIp', 'probeS4', 'probeR9',
            'probePhiWidth', 'probeSigmaIeIe', 'probeEtaWidth']
          iso:
            ch:
              ['probeChIso03', 'probeChIso03worst']
            ph:
              ['probePhoIso']

        quantiles:
          [0.01,0.05,0.1,0.15,0.2,0.25,
          0.3,0.35,0.4,0.45,0.5,0.55,
          0.6,0.65,0.7,0.75,0.8,0.85,
          0.9,0.95,0.99]

        n_events:
          1000

        work_dir:
          path/to/work_dir

        weights_dir:
          path/to/weights_dir


    Attributes:
        cluster (distributed.deploy.local.LocalCluster): Dask cluster used to
            distribute the computation; be default a LocalCluster (single machine)
            is used
        client (distributed.client.Client): Dask client which connects to the cluster
        quantiles (list): list of quantiles used to build all the quantile regression
            chain objects
        n_events (int): amount of events (i.e. number of rows) loaded from a pandas
            dataframe when methods like loadDataDF/loadMCDF etc. are called
        work_dir (str): path to the working directory where the dataframes are
            expected to be stored
        weights_dir (str): path to the directory where the serialized trained
            regressors are saved; if relative, work_dir is prepended
        qrcs (defaultdict): nested dictionary containing the quantile regressors
            created with the information contained in the config file
            Ex:

            qrcs = {
                'data': {
                    'EB': {
                        'SS': QRC()
                        'iso': {
                            'ch': QRCd(),
                            'ph': QRCd()
                        }
                    }.
                    'EE': {
                        'SS': QRC()
                        'iso': {
                            'ch': QRCd(),
                            'ph': QRCd()
                        }
                    }.
                },
                'mc': {
                    'EB': {
                        'SS': QRC()
                        'iso': {
                            'ch': QRCd(),
                            'ph': QRCd()
                        }
                    }.
                    'EE': {
                        'SS': QRC()
                        'iso': {
                            'ch': QRCd(),
                            'ph': QRCd()
                        }
                    }.
                },
            }

            To loop over the qrcs, obtain an iterator object by feeding this
            object to the generator 'scan_dict'
        paths (defaultdict): nested dictionary which mimics the structure of
            qrcs but contains a list with [dataset, detector, var_type, (iso_type)]
            instead; its main use is with zip() (after obtaining an iterator from
            from scan_dict) and the iterator obtained from qrcs, in order to
            associate the information to every qrc object
    """
    def __init__(self, config_file_name):
        # Setup default Dask local cluster
        self.cluster = LocalCluster()
        logger.info('Setting up {}'.format(self.cluster))
        self.client = Client(self.cluster)
        logger.info('Setting up Client {}'.format(self.client))

        # Get information from yaml config file
        stream = open(config_file_name, 'r')
        input_dict = yaml.safe_load(stream)
        self.dataframes = input_dict['dataframes']
        year = input_dict['year']
        datasets = input_dict['datasets']
        detectors = input_dict['detectors']
        variables = input_dict['variables']
        self.quantiles = input_dict['quantiles']
        if 'n_events' in input_dict:
            self.n_events = input_dict['n_events']
        else:
            self.n_events = -1

        self.work_dir = input_dict['work_dir']
        self.weights_dir = input_dict['weights_dir']

        # Create dictionary for qrcs
        self.qrcs = rec_dd()
        self.paths = rec_dd()
        for dataset in datasets:
            for detector in detectors:
                for var_type, vals in variables.items():
                    if var_type == 'SS':
                        self.qrcs[dataset][detector][var_type] = QRC(
                                year, detector, self.work_dir, vals, self.quantiles)
                        self.paths[dataset][detector][var_type] = [
                                dataset, detector, var_type
                                ]
                    elif var_type == 'iso':
                        for iso_type, iso_variables in vals.items():
                            self.qrcs[dataset][detector][var_type][iso_type] = QRCd(
                                    year, detector, self.work_dir, iso_variables, self.quantiles)
                            self.paths[dataset][detector][var_type][iso_type] = [
                                    dataset, detector, var_type, iso_type
                                    ]

    def load_dataframes(self):
        """
        Call either LoadDataDF or LoadMCDF for every QRC/QRCd object created
        """
        default_columns = ['probePt', 'probeScEta', 'probePhi', 'rho']
        for qrc, path in zip(scan_dict(self.qrcs), scan_dict(self.paths)):
            logger.debug(path)
            dataset = path[0]
            detector = path[1]
            var_type = path[2]
            if dataset == 'data':
                qrc.loadDataDF(
                        self.dataframes[dataset][detector][var_type], 0,
                        self.n_events, rsh=False, columns=default_columns+qrc.vars
                        )
            else:
                qrc.loadMCDF(
                        self.dataframes[dataset][detector][var_type], 0,
                        self.n_events, rsh=False, columns=default_columns+qrc.vars
                        )

    def train_regressors(self):
        """Train the regressors created when creating a QRCScheduler object by calling
        either trainAllData or trainAllMC for each of them (depending on the datadet).
        N.B.: data-like regressors can be trained independently from one another,
        so we append all the futures returned by trainAllData to a futures list;
        mc-like regressors can be parallelized but each of them has operations
        executed in a specific order inside, so the parallelization is done in this
        part with the client.submit function
        """
        self.data_futures = []
        self.mc_futures = []

        for qrc, path in zip(scan_dict(self.qrcs), scan_dict(self.paths)):
            dataset = path[0]
            if dataset == 'data':
                self.data_futures.append(qrc.trainAllData(self.weights_dir, self.client))
        logger.info('Submitting data training')
        progress(self.data_futures)
        wait(self.data_futures)

        for qrc, path in zip(scan_dict(self.qrcs), scan_dict(self.paths)):
            dataset = path[0]
            if dataset == 'mc':
                self.mc_futures.append(self.client.submit(qrc.trainAllMC, self.weights_dir))
        logger.info('Submitting MC training')
        progress(self.mc_futures)
        wait(self.mc_futures)

    def setup_slurm_cluster(self, config_file):
        """ Setup cluster to distribute the computation of the regressors.
        If config_file is None, a local cluster is setup (i.e., we run locally).
        If config_file is a yaml file in the form:
        (...)
        jobqueue:
          slurm:
            cores: 72
            memory: 100GB
            jobs: 100
            (...)
        (...)
        a SLURM cluster is setup with the information passed.
        See https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html#dask_jobqueue.SLURMCluster
        for more.
        Args:
            config_file (str): name of the configuration file to setup the SLURM cluster
        """

        # Close default local cluster and disconnect client
        self.client.close()
        self.cluster.close()
        logger.info('Closing default Client')
        logger.info('Closing default LocalCluster')
        del self.cluster
        del self.client

        # Open new SLURM cluster and connect client
        stream = open(config_file, 'r')
        inp = yaml.safe_load(stream)
        cores = inp['jobqueue']['slurm']['cores']
        memory = inp['jobqueue']['slurm']['memory']
        jobs = inp['jobqueue']['slurm']['jobs']
        if not os.path.isdir('slurm_logs'):
            os.makedirs('slurm_logs')
        if 'queue' in inp['jobqueue']['slurm']:
            queue = inp['jobqueue']['slurm']['queue']
            self.cluster = SLURMCluster(
                    cores=cores,
                    memory=memory,
                    queue=queue,
                    log_directory='slurm_logs'
                    )
        else:
            self.cluster = SLURMCluster(
                    cores=cores,
                    memory=memory,
                    log_directory='slurm_logs',
                    walltime='10:00:00'
                    )
        #self.cluster.adapt(maximum_jobs = jobs)
        self.cluster.scale(jobs)
        self.client = Client(self.cluster)
        logger.info('Setting up {}'.format(self.cluster))
        logger.info('Setting up Client {}'.format(self.client))

    def connect_to_cluster(self, cluster_id):
        # Close default local cluster and disconnect client
        self.client.close()
        self.cluster.close()
        logger.info('Closing default Client')
        logger.info('Closing default LocalCluster')
        del self.cluster
        del self.client

        self.client = Client(cluster_id)
        logger.info('Setting up Client {}'.format(self.client))
