"""Example of how to add MemGPT into an AutoGen groupchat

Based on the official AutoGen example here: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb

Begin by doing:
  pip install "pyautogen[teachable]"
  pip install pymemgpt
  or
  pip install -e . (inside the MemGPT home directory)
"""


import os
import autogen
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
from memgpt.presets.presets import DEFAULT_PRESET
from memgpt.constants import LLM_MAX_TOKENS


# LLM_BACKEND = "openai"
LLM_BACKEND = "azure"
# LLM_BACKEND = "local"

if LLM_BACKEND == "openai":
    # For demo purposes let's use gpt-4
    model = "gpt-4-32k"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "You must set OPENAI_API_KEY to run this example"

    # This config is for AutoGen agents that are not powered by MemGPT
    config_list = [
        {
            "model": model,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]

    # This config is for AutoGen agents that powered by MemGPT
    config_list_memgpt = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # OpenAI specific
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "openai_key": openai_api_key,
        },
    ]

elif LLM_BACKEND == "azure":
    # Make sure that you have access to this deployment/model on your Azure account!
    # If you don't have access to the model, the code will fail
    model = "gpt-4"

    azure_openai_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert (
        azure_openai_api_key is not None and azure_openai_version is not None and azure_openai_endpoint is not None
    ), "Set all the required OpenAI Azure variables (see: https://memgpt.readthedocs.io/en/latest/endpoints/#azure)"

    # This config is for AutoGen agents that are not powered by MemGPT
    config_list = [
        {
            "model": model,
            "api_type": "azure",
            "api_key": azure_openai_api_key,
            "api_version": azure_openai_version,
            # NOTE: on versions of pyautogen < 0.2.0, use "api_base"
            # "api_base": azure_openai_endpoint,
            "base_url": azure_openai_endpoint,
        }
    ]

    # This config is for AutoGen agents that powered by MemGPT
    config_list_memgpt = [
        {
            "model": model,
            "context_window": 32768, #LLM_MAX_TOKENS[model],
            "preset": 'memgpt_workato',
            "model_wrapper": None,
            # Azure specific
            "model_endpoint_type": "azure",
            "model_endpoint": "https://customproject.openai.azure.com",
            "azure_key": azure_openai_api_key,
            "azure_endpoint": azure_openai_endpoint,
            "azure_version": azure_openai_version,
        },
    ]

elif LLM_BACKEND == "local":
    # Example using LM Studio on a local machine
    # You will have to change the parameters based on your setup

    # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            # NOTE: on versions of pyautogen < 0.2.0 use "api_base", and also uncomment "api_type"
            # "api_base": "http://localhost:1234/v1",
            # "api_type": "open_ai",
            "base_url": "http://localhost:1234/v1",  # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
            "api_key": "NULL",  #  not needed
        },
    ]

    # MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_memgpt = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,  # only required for Ollama, see: https://memgpt.readthedocs.io/en/latest/ollama/
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
            "model_wrapper": "airoboros-l2-70b-2.1",  # airoboros is the default wrapper and should work for most models
            "model_endpoint_type": "lmstudio",  # can use webui, ollama, llamacpp, etc.
            "model_endpoint": "http://localhost:1234",  # the IP address of your LLM backend
        },
    ]

else:
    raise ValueError(LLM_BACKEND)

# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = True

# Set to True if you want to print MemGPT's inner workings.
DEBUG = True

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": True,
    "show_function_outputs": True,
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

print("--llm_config--")
print(llm_config)
print("--llm_config_memgpt--")
print(llm_config_memgpt)
print("")


def is_termination_msg(data):
    has_content = "content" in data and data["content"] is not None
    return has_content and "TERMINATE" in data["content"]


analyst_system_message = f"""

Workato Analyst: As a Senior Workato Analyst responsible for a Workato job log analysis.

Follow the plan strictily:
1. Request Workato Job Logs from MemGPT_coder without engaging in coding tasks; your role is solely to obtain the information from MemGPT_coder.
2. Wait until MemGPT_coder completes the response.
3. Analyze the job log data, provide comments, and send the Workato Job Logs to the UI designer for chart creation upon receiving the response from MemGPT_coder.
4. Upon receiving the code from the UI designer along with the complete dataset, if it's not provided, kindly request the UI designer to provide the missing data.
5. Execute the Python code received from the UI designer on your local machine to generate and display the chart.

Upon successful completion of the plan, add "TERMINATE" to conclude the message.

"""

analyst = autogen.AssistantAgent(
    name = "analyst",
    system_message = analyst_system_message,
    llm_config=llm_config,
    is_termination_msg=is_termination_msg,
    code_execution_config={"work_dir": "coding"}
)

uidesigner_system_message = f"""

UI Designer: You are a Senior UI/UX designer with a specialization in crafting charts with Python. 
Your task is to create a chart using the data provided by the Workato Analyst.

Follow the plan strictily:
1. Producing thorough code that incorporates the entire, actual dataset.
2. Verifying that the code is ready for immediate execution, devoid of any placeholder text or missing data.


"""

uidesigner = autogen.AssistantAgent(
    name = "uidesigner",
    system_message=uidesigner_system_message,
    code_execution_config=False,  # set to True or image name like "python:3" to use docker
    llm_config=llm_config
)

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="admin",
    system_message="Human Admin: Let's engage with the analyst to have a discussion about the Workato job report.",
    code_execution_config=False,
    human_input_mode="NEVER", 
    is_termination_msg=is_termination_msg
)

if not USE_MEMGPT:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    engineer = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )

else:
    # In our example, we swap this AutoGen agent with a MemGPT agent
    # This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
    engineer_system_message = f"""
    As MemGPT_coder, you function as a Senior Workato Engineer responsible for executing the getWorkatoRecipe internal function as specified by the Workato Analyst.

    You are participating in a group chat with a user ({user_proxy.name}) and a analyst ({analyst.name}).

    Follow the above plan strictily.

    """

    MemGPT_coder = create_memgpt_autogen_agent_from_config(
        "MemGPT_coder",
        llm_config=llm_config_memgpt,
        system_message=engineer_system_message,
        default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
    )

# Initialize the group chat between the user and two LLM agents (PM and coder)
# groupchat = autogen.GroupChat(agents=[user_proxy, pm, coder], messages=[], max_round=12)
groupchat = autogen.GroupChat(agents=[user_proxy, analyst, uidesigner, MemGPT_coder], messages=[], max_round=20)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
message = f"""
Hello, Workato developer! Start by greeting me. On your second interaction, provide a list of the Workato recipes with folder id 2016066 and inform me of the results",
"""

user_proxy.initiate_chat(manager, clear_history=True, message=message)
