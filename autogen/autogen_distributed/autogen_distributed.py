import asyncio
import logging
import os
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core._serialization import DataclassJsonMessageSerializer
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime, GrpcWorkerAgentRuntimeHost
from autogen_ext.tools.langchain import LangChainToolAdapter
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HOST_ADDRESS = "localhost:50051"
MODEL = "gpt-4o-mini"
ALL_IN_ONE_WORKER = False

PROS_PROMPT = (
    "To help with a decision on whether to use AutoGen in a new AI Agent project, "
    "please research and briefly respond with reasons in favor of choosing AutoGen; the pros of AutoGen."
)
CONS_PROMPT = (
    "To help with a decision on whether to use AutoGen in a new AI Agent project, "
    "please research and briefly respond with reasons against choosing AutoGen; the cons of AutoGen."
)
JUDGE_PROMPT = (
    "You must make a decision on whether to use AutoGen for a project. "
    "Your research team has come up with the following reasons for and against. "
    "Based purely on the research from your team, respond with your decision and brief rationale."
)


@dataclass
class Message:
    content: str


def make_search_tool() -> LangChainToolAdapter:
    serper = GoogleSerperAPIWrapper()
    lc_tool = Tool(
        name="internet_search",
        func=serper.run,
        description="Useful for when you need to search the internet",
    )
    return LangChainToolAdapter(lc_tool)


class ResearchAgent(RoutedAgent):
    def __init__(self, name: str, search_tool: LangChainToolAdapter) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model=MODEL)
        self._delegate = AssistantAgent(
            name, model_client=model_client, tools=[search_tool], reflect_on_tool_use=True
        )

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        return Message(content=response.chat_message.content)


class JudgeAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model=MODEL)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        pros, cons = await asyncio.gather(
            self.send_message(Message(content=PROS_PROMPT), AgentId("researcher_pros", "default")),
            self.send_message(Message(content=CONS_PROMPT), AgentId("researcher_cons", "default")),
        )

        research = (
            f"## Pros of AutoGen:\n{pros.content}\n\n"
            f"## Cons of AutoGen:\n{cons.content}\n\n"
        )
        verdict_prompt = f"{JUDGE_PROMPT}\n\n{research}Respond with your decision and brief explanation."
        response = await self._delegate.on_messages(
            [TextMessage(content=verdict_prompt, source="user")], ctx.cancellation_token
        )
        return Message(content=research + "## Decision:\n\n" + response.chat_message.content)


def _check_env() -> None:
    missing = [v for v in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def _register_serializers(worker: GrpcWorkerAgentRuntime) -> None:
    worker.add_message_serializer(DataclassJsonMessageSerializer(Message))


async def main() -> None:
    _check_env()
    search_tool = make_search_tool()
    host = GrpcWorkerAgentRuntimeHost(address=HOST_ADDRESS)
    host.start()
    logger.info("Host started at %s", HOST_ADDRESS)

    workers: list[GrpcWorkerAgentRuntime] = []
    try:
        if ALL_IN_ONE_WORKER:
            worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
            await worker.start()
            _register_serializers(worker)
            workers.append(worker)
            await ResearchAgent.register(worker, "researcher_pros", lambda: ResearchAgent("researcher_pros", search_tool))
            await ResearchAgent.register(worker, "researcher_cons", lambda: ResearchAgent("researcher_cons", search_tool))
            await JudgeAgent.register(worker, "judge", lambda: JudgeAgent("judge"))
            judge_worker = worker
        else:
            worker1 = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
            await worker1.start()
            _register_serializers(worker1)
            workers.append(worker1)
            await ResearchAgent.register(worker1, "researcher_pros", lambda: ResearchAgent("researcher_pros", search_tool))

            worker2 = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
            await worker2.start()
            _register_serializers(worker2)
            workers.append(worker2)
            await ResearchAgent.register(worker2, "researcher_cons", lambda: ResearchAgent("researcher_cons", search_tool))

            judge_worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
            await judge_worker.start()
            _register_serializers(judge_worker)
            workers.append(judge_worker)
            await JudgeAgent.register(judge_worker, "judge", lambda: JudgeAgent("judge"))

        response = await judge_worker.send_message(Message(content="Go!"), AgentId("judge", "default"))
        print(response.content)

    finally:
        for w in workers:
            await w.stop()
        await host.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
