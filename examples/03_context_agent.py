from typing import Any
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, cerebras, deepgram

load_dotenv()

server = AgentServer()


class Assistant(Agent):
    def __init__(self, user_context: dict[str, Any] = None) -> None:
        SYSTEM_PROMPT = "You are a helpful assistant. The user name is {name}. They are {age} years old and work as a {job} in {location}."
        if user_context:
            SYSTEM_PROMPT = SYSTEM_PROMPT.format(**user_context)

        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()


@server.rtc_session(agent_name="context-agent")
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

    user_context = {
        "name": "Lokesh",
        "age": 25,
        "job": "Software Engineer",
        "location": "Bangalore, India",
    }

    agent = Assistant(user_context=user_context)

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
