import os
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient, Input, command, Output, dsl
from azure.ai.ml.entities import JobResourceConfiguration, Environment

subscription_id = os.getenv("AML_SUBSCRIPTION")
resource_group = os.getenv("AML_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
GPU_TYPE = "Singularity.ND96amrs_A100_v4"
CLUSTER = "/subscriptions/86956e13-ab30-4f91-945c-422273bae7bf/resourceGroups/singularity-aiplatform/providers/Microsoft.MachineLearningServices/virtualClusters/nonipp-shared"

credential = AzureCliCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)


def load_component(component_name):
    component = ml_client.components.get(component_name)
    print(f"Loaded component {component.name} and version {component.version}")
    return component

test_component = load_component("simplerl_reason")

def submit_job():
    @dsl.pipeline(
        compute=CLUSTER,
        description="simplerl",
        experiment_name="simplerl_reason_math",
    )
    def anonymous_pipeline():
        output_dict = {}
        test_job = test_component(n=3)
        test_job.name = f"mathlvl35_{test_job.name}"
        test_job.resources = JobResourceConfiguration(instance_count=1, instance_type=GPU_TYPE)
        output_dict["test_outputs"] = test_job.outputs.output_dir
        return output_dict

    actual_pipeline = anonymous_pipeline()
    returned_job = ml_client.jobs.create_or_update(actual_pipeline)
    print(returned_job.studio_url)

if __name__ == "__main__":
    submit_job()