from typing import Any
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, cerebras, deepgram
import requests

load_dotenv()

server = AgentServer()

BASE_URL = "https://api.frankfurter.dev/v2"


def get_exchange_rate(base: str, quote: str) -> float:
    url = f"{BASE_URL}/rate/{base.upper()}/{quote.upper()}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()
    return float(data["rate"])


class ToolCalling(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. You can use the convert_currency tool to convert currencies.",
        )

    @function_tool
    async def convert_currency(
        self, amount: float, base: str, quote: str
    ) -> dict[str, Any]:
        """
        Convert a currency to another currency.

        Args:
            amount: The amount to convert.
            base: The base currency.
            quote: The quote currency.

        Returns:
            A dictionary containing the converted amount, the base currency, the quote currency, the rate, and the converted amount.
        """
        rate = get_exchange_rate(base, quote)
        converted = round(amount * rate, 2)

        return {
            "amount": amount,
            "from": base.upper(),
            "to": quote.upper(),
            "rate": rate,
            "converted_amount": converted,
        }

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")


@server.rtc_session(agent_name="tool-calling")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=cerebras.LLM(model="gemma-4-31b"),
        tts=inference.TTS(
            model="cartesia/sonic-3.5", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )
    agent = ToolCalling()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L
                ),
            ),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
