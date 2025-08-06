from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.identity import AzureCliCredential
import os
from azure.ai.ml import MLClient

credential = AzureCliCredential()
subscription_id = os.getenv("AML_SUBSCRIPTION")
resource_group = os.getenv("AML_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

def make_test_command(ml_client, environment):
    COMMAND = (
        """bash ty373.sh"""
    )

    component = CommandComponent(
        name="simplerl_reason",
        display_name="simplerl",
        description="simplerl reason w math",
        inputs={
            "n": {"type": "number", "default": -1},
        },
        outputs={
            "output_dir": Output(data_type="uri_folder")
        },
        code=".",
        command=COMMAND,
        environment=environment,
        is_deterministic=False
    )
    created_component = ml_client.components.create_or_update(component)
    print(created_component.name, created_component.version)

if __name__ == "__main__":
    make_test_command(ml_client, "goji:11")


