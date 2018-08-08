from rastervision.workflows.config_utils import (
    make_model_config, CL, OD)
from rastervision.workflows.chain_utils import (
    ChainWorkflowPaths, ChainWorkflowSceneGenerator,
    ChainWorkflow)


def main():
    base_uri = '/opt/data/lf-dev/workflow-dreams/'
    task = OD
    backend_config_uri = ''
    pretrained_model_uri = ''
    paths = ChainWorkflowPaths(base_uri)
    model_config = make_model_config(['car'], task)
    scene_generator = ChainWorkflowSceneGenerator(paths)

    def make_scene(id, raster_uris, ground_truth_labels_uri):
        return scene_generator.make_geotiff_geojson_scene(
            id, raster_uris, task,
            ground_truth_labels_uri=ground_truth_labels_uri)

    train_scenes = [
        make_scene('2-10', ['/test/2-10.tif'], '/test/2-10.json'),
        make_scene('2-11', ['/test/2-11.tif'], '/test/2-11.json')
    ]
    validation_scenes = [
        make_scene('2-12', ['/test/2-12.tif'], '/test/2-12.json')
    ]

    workflow = ChainWorkflow(
        paths, model_config, train_scenes, validation_scenes,
        backend_config_uri, pretrained_model_uri)
    workflow.save_config()


if __name__ == '__main__':
    main()
