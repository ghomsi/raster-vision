from os.path import join
import copy

from rastervision.workflows.config_utils import (
    make_geotiff_geojson_scene, make_compute_stats,
    make_make_chips, make_train,
    make_predict, make_predict, make_eval, make_stats_uri,
    ClassificationGeoJSONOptions, OD, CL)
from rastervision.utils.files import save_json_config
from rastervision.protos.chain_workflow_pb2 import ChainWorkflowConfig

COMPUTE_STATS = 'compute-stats'
MAKE_CHIPS = 'make-chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'


class ChainWorkflowPaths():
    def __init__(self, base_uri, compute_stats_uri=None, make_chips_uri=None,
                 train_uri=None, predict_uri=None, eval_uri=None,
                 workflow_uri=None):
        self.compute_stats_uri = (compute_stats_uri if compute_stats_uri else join(base_uri, COMPUTE_STATS))
        self.make_chips_uri = (make_chips_uri if make_chips_uri else join(base_uri, MAKE_CHIPS))
        self.train_uri = (train_uri if train_uri else join(base_uri, TRAIN))
        self.predict_uri = (predict_uri if predict_uri else join(base_uri, PREDICT))
        self.eval_uri = (eval_uri if eval_uri else join(base_uri, EVAL))
        self.workflow_uri = (workflow_uri if workflow_uri else join(base_uri, 'workflow.json'))

    def get_config_uri(self, command, uri):
        return join(uri, command + '-config.json')


class ChainWorkflowSceneGenerator():
    def __init__(self, paths, chip_size=300):
        self.paths = paths
        self.chip_size = chip_size

    def make_geotiff_geojson_scene(
            self,
            id,
            raster_uris,
            task_options,
            ground_truth_labels_uri=None,
            channel_order=[0, 1, 2]):
        if type(task_options) is int:
            if task_options == OD:
                task_options = None
            elif task_options == CL:
                task_options = ClassificationGeoJSONOptions()
            else:
                raise ValueError('Unknown task: ' + task_options)

        if task_options is not None:
            if type(task_options) is ClassificationGeoJSONOptions:
                # Force cell_size to be consistent with rest of chain workflow.
                task_options = copy.deepcopy(task_options)
                task_options.cell_size = self.chip_size
            else:
                raise ValueError('Unknown type of task_options: ' +
                                 str(type(task_options)))

        return make_geotiff_geojson_scene(
            id,
            raster_uris,
            make_stats_uri(self.paths.compute_stats_uri),
            task_options,
            ground_truth_labels_uri=ground_truth_labels_uri,
            prediction_base_uri=self.paths.predict_uri,
            channel_order=channel_order)


class ChainWorkflow(object):
    def __init__(
            self,
            paths,
            model_config,
            train_scenes,
            validation_scenes,
            backend_config_uri,
            pretrained_model_uri,
            sync_interval=600,
            test_scenes=None,
            chip_size=300,
            debug=True,
            task_make_chips_options=None,
            task_predict_options=None):
        self.paths = paths
        self.model_config = model_config
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.backend_config_uri = backend_config_uri
        self.pretrained_model_uri = pretrained_model_uri
        self.sync_interval = sync_interval
        self.test_scenes = [] if test_scenes is None else test_scenes
        self.chip_size = chip_size
        self.debug = debug
        self.task_make_chips_options = (
            model_config.task if task_make_chips_options is None
            else task_make_chips_options)
        self.task_predict_options = (
            model_config.task if task_predict_options is None
            else task_predict_options)

    def make_compute_stats(self):
        scenes = self.train_scenes + self.validation_scenes + self.test_scenes
        raster_sources = [scene.raster_source for scene in scenes]
        return make_compute_stats(raster_sources, self.paths.compute_stats_uri)

    def make_make_chips(self):
        return make_make_chips(
            self.train_scenes,
            self.validation_scenes,
            self.model_config,
            self.paths.make_chips_uri,
            chip_size=self.chip_size,
            debug=self.debug,
            task_options=self.task_make_chips_options)

    def make_train(self):
        return make_train(
            self.model_config,
            self.backend_config_uri,
            self.paths.make_chips_uri,
            self.paths.train_uri,
            self.pretrained_model_uri,
            self.sync_interval)

    def make_predict(self):
        return make_predict(
            self.model_config,
            self.validation_scenes + self.test_scenes,
            self.chip_size,
            self.paths.train_uri,
            self.paths.predict_uri,
            debug=self.debug,
            task_options=self.task_predict_options)

    def make_eval(self):
        return make_eval(
            self.model_config,
            self.validation_scenes,
            self.paths.eval_uri,
            debug=self.debug)

    def save_config(self):
        config = ChainWorkflowConfig()

        config.model_config.MergeFrom(self.model_config)
        config.chip_size = self.chip_size
        config.train_scenes.extend(self.train_scenes)
        config.validation_scenes.extend(self.validation_scenes)
        config.test_scenes.extend(self.test_scenes)

        config.compute_stats_uri = self.paths.compute_stats_uri
        config.make_chips_uri = self.paths.make_chips_uri
        config.train_uri = self.paths.train_uri
        config.predict_uri = self.paths.predict_uri
        config.eval_uri = self.paths.eval_uri

        config.compute_stats.MergeFrom(self.make_compute_stats())
        config.make_chips.MergeFrom(self.make_make_chips())
        config.train.MergeFrom(self.make_train())
        config.predict.MergeFrom(self.make_predict())
        config.eval.MergeFrom(self.make_eval())

        save_json_config(config, self.paths.workflow_uri)
        print('Wrote workflow config to: ' + self.paths.workflow_uri)
