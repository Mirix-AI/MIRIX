from mirix.orm.step import Step as StepModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.step import Step as PydanticStep
from mirix.utils import enforce_types


class StepManager:
    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def log_step(
        self,
        actor: PydanticClient,
        provider_name: str,
        model: str,
        context_window_limit: int,
        usage: UsageStatistics,
    ) -> PydanticStep:
        step_data = {
            "origin": None,
            "organization_id": actor.organization_id,
            "provider_name": provider_name,
            "model": model,
            "context_window_limit": context_window_limit,
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "tags": [],
            "tid": None,
        }
        async with self.session_maker() as session:
            new_step = StepModel(**step_data)
            await new_step.create(session)
            return new_step.to_pydantic()

    @enforce_types
    async def get_step(self, step_id: str) -> PydanticStep:
        async with self.session_maker() as session:
            step = await StepModel.read(db_session=session, identifier=step_id)
            return step.to_pydantic()
