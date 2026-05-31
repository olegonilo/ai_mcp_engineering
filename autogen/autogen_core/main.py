import asyncio
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Message:
    content: str


# --- Demo 1: Simple echo agent ---

class SimpleAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Simple")

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"This is {self.id.type}-{self.id.key}. You said '{message.content}' and I disagree.")


async def demo_simple_agent() -> None:
    runtime = SingleThreadedAgentRuntime()
    await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())
    runtime.start()

    response = await runtime.send_message(Message("Well hi there!"), AgentId("simple_agent", "default"))
    print(">>> [SimpleAgent]", response.content)

    await runtime.stop()
    await runtime.close()


# --- Demo 2: LLM agent chained with simple agent ---

class MyLLMAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__("LLMAgent")
        self._delegate = AssistantAgent("LLMAgent", model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        print(f"{self.id.type} received message: {message.content}")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        reply = response.chat_message.content
        print(f"{self.id.type} responded: {reply}")
        return Message(content=reply)


async def demo_llm_chain() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    runtime = SingleThreadedAgentRuntime()
    try:
        await SimpleAgent.register(runtime, "simple_agent", lambda: SimpleAgent())
        await MyLLMAgent.register(runtime, "LLMAgent", lambda: MyLLMAgent(model_client))
        runtime.start()

        response = await runtime.send_message(Message("Hi there!"), AgentId("LLMAgent", "default"))
        print(">>> [LLMAgent]", response.content)

        response = await runtime.send_message(Message(response.content), AgentId("simple_agent", "default"))
        print(">>> [SimpleAgent]", response.content)

        response = await runtime.send_message(Message(response.content), AgentId("LLMAgent", "default"))
        print(">>> [LLMAgent]", response.content)

        await runtime.stop()
        await runtime.close()
    finally:
        await model_client.close()


# --- Demo 3: Rock Paper Scissors (GPT-4o-mini vs GPT-4o-mini) ---

JUDGE_PROMPT = "You are judging a game of rock, paper, scissors. The players have made these choices:\n"
RPS_INSTRUCTION = "You are playing rock, paper, scissors. Respond only with one word: rock, paper, or scissors."


class Player1Agent(RoutedAgent):
    def __init__(self, name: str, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)


class Player2Agent(RoutedAgent):
    def __init__(self, name: str, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)


class RockPaperScissorsAgent(RoutedAgent):
    def __init__(self, name: str, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        prompt = Message(content=RPS_INSTRUCTION)
        response1 = await self.send_message(prompt, AgentId("player1", "default"))
        response2 = await self.send_message(prompt, AgentId("player2", "default"))

        result = f"Player 1: {response1.content}\nPlayer 2: {response2.content}\n"
        judgement = TextMessage(content=f"{JUDGE_PROMPT}{result}Who wins?", source="user")
        verdict = await self._delegate.on_messages([judgement], ctx.cancellation_token)
        return Message(content=result + verdict.chat_message.content)


async def demo_rock_paper_scissors() -> None:
    player1_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
    player2_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
    judge_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    runtime = SingleThreadedAgentRuntime()
    try:
        await Player1Agent.register(runtime, "player1", lambda: Player1Agent("player1", player1_client))
        await Player2Agent.register(runtime, "player2", lambda: Player2Agent("player2", player2_client))
        await RockPaperScissorsAgent.register(
            runtime, "rock_paper_scissors",
            lambda: RockPaperScissorsAgent("rock_paper_scissors", judge_client)
        )
        runtime.start()

        response = await runtime.send_message(Message("go"), AgentId("rock_paper_scissors", "default"))
        print(">>> [RPS]", response.content)

        await runtime.stop()
        await runtime.close()
    finally:
        await player1_client.close()
        await player2_client.close()
        await judge_client.close()


async def main() -> None:
    print("=== Demo 1: Simple Agent ===")
    await demo_simple_agent()

    print("\n=== Demo 2: LLM Chain ===")
    await demo_llm_chain()

    print("\n=== Demo 3: Rock Paper Scissors ===")
    await demo_rock_paper_scissors()


if __name__ == "__main__":
    asyncio.run(main())
