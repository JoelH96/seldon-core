
import kfp.dsl as dsl
import yaml
from kubernetes import client as k8s


@dsl.pipeline(
  name='Income Classifier',
  description='A pipeline demonstrating reproducible steps for NLP'
)
def income_pipeline(
        tabular_data="/mnt/tabular_data.data",
	preprocessor_path="/mnt/preprocessor.model",
        model_path="/mnt/income_class.model",
        out_path="/mnt/clf_prediction.data"):
    """
    Pipeline 
    """
    vop = dsl.VolumeOp(
        name='my-pvc',
        resource_name="my-pvc",
        modes=["ReadWriteMany"],
        storage_class="nfs-client",
        size="1Gi"
    )

    predict_step = dsl.ContainerOp(
        name='predictor',
        image='gcr.io/dev-joel/income_classifier:0.1',
	    command="python",
        arguments=[
            "/microservice/pipeline_step.py",
	    "--tabular-data", tabular_data,
            "--preprocessor-path", preprocessor_path,
            "--model-path", model_path,
            "--out-path", out_path,
            "--action", "train",
        ],
        pvolumes={"/mnt": vop.volume}
    )

    try:
        seldon_config = yaml.load(open("../deploy_pipeline/seldon_production_pipeline.yaml"), Loader=yaml.FullLoader)
    except:
        # If this file is run from the project core directory 
        seldon_config = yaml.load(open("deploy_pipeline/seldon_production_pipeline.yaml"), Loader=yaml.FullLoader)

    deploy_step = dsl.ResourceOp(
        name="seldondeploy",
        k8s_resource=seldon_config,
        attribute_outputs={"name": "{.metadata.name}"})

    deploy_step.after(predict_step)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(income_pipeline, __file__ + '.tar.gz')
